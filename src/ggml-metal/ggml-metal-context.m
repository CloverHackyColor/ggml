#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8
#define GGML_METAL_MAX_PREFETCH_STAGES 8

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    // shared device queue, or an owned one for secondary (prefetch) contexts
    // so their uploads overlap the primary context's compute
    id<MTLCommandQueue> queue;
    bool owns_queue;
    bool is_prefetch;

    // Bounded host-visible source ring for expert uploads. Keeping one no-copy
    // MTLBuffer per CPU tensor can pin many GiB of host pages on discrete GPUs.
    id<MTLBuffer>        prefetch_stages[GGML_METAL_MAX_PREFETCH_STAGES];
    id<MTLCommandBuffer> prefetch_stage_cmds[GGML_METAL_MAX_PREFETCH_STAGES];
    int prefetch_stage_count;
    int prefetch_stage_cur;

    ggml_metal_event_t ev_cpy; // for async copies

    // Synchronization and resources shared by both cross-device copy paths.
    struct ggml_metal_xdev_link {
        ggml_metal_device_t  src_dev;
        void *               host;
        size_t               cap;
        id<MTLBuffer>        wrap_src, wrap_dst;
        // Retaining the source keeps its cached remote view valid.
        id<MTLBuffer>        peer_src, peer_view;
        id<MTLSharedEvent>   ev_ready, ev_done;
        uint64_t             seq;
        bool                 peer_logged;
    } xlinks[4];
    int n_xlinks;

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_fusion;
    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;
    int debug_fusion;

    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;
};

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev, bool prefetch) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    // TODO: would it be better to have one queue for the backend and one queue for the device?
    //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
    //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
    res->queue = prefetch ? ggml_metal_device_acquire_queue(dev, &res->owns_queue)
                          : ggml_metal_device_get_queue(dev);
    if (res->queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        free(res);
        return NULL;
    }
    res->is_prefetch = prefetch;
    if (prefetch) {
        const char * val = getenv("GGML_METAL_PREFETCH_STAGE_COUNT");
        const int requested = val ? atoi(val) : 1;
        res->prefetch_stage_count = MAX(1, MIN(requested, GGML_METAL_MAX_PREFETCH_STAGES));
    }

    res->dev = dev;
    res->lib = ggml_metal_device_get_library(dev);
    if (res->lib == NULL) {
        GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
        GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

        res->lib = ggml_metal_library_init(dev);
        if (res->lib == NULL) {
            GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

            free(res);

            return NULL;
        }
    }

    res->ev_cpy = ggml_metal_device_event_init(dev);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_fusion      = getenv("GGML_METAL_FUSION_DISABLE") == nil;

    // discrete GPUs (AMD) corrupt their output with a concurrent encoder: the driver does
    // not honour the memory barrier. GGML_METAL_CONCURRENCY_ENABLE forces it back on to test.
    res->use_concurrency = props_dev->has_unified_memory
        ? getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil
        : getenv("GGML_METAL_CONCURRENCY_ENABLE")  != nil;

    {
        const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
        res->debug_graph = val ? atoi(val) : 0;
    }

    {
        const char * val = getenv("GGML_METAL_FUSION_DEBUG");
        res->debug_fusion = val ? atoi(val) : 0;
    }

    res->use_graph_optimize = true;

    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
    GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
    GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

    res->capture_compute = 0;
    res->capture_started = false;
    res->capture_scope = nil;

    {
        const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
        if (val) {
            res->capture_compute = atoi(val);
        }
    }

    res->has_error = false;

    res->gf = nil;
    res->encode_async = nil;
    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->cmd_bufs[i].obj = nil;
    }

    res->cmd_bufs_ext = [[NSMutableArray alloc] init];

    res->cmd_buf_last = nil;

    res->pipelines_ext = ggml_metal_pipelines_init();

    return res;
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < ctx->prefetch_stage_count; ++i) {
        if (ctx->prefetch_stage_cmds[i]) {
            [ctx->prefetch_stage_cmds[i] waitUntilCompleted];
            [ctx->prefetch_stage_cmds[i] release];
        }
        [ctx->prefetch_stages[i] release];
    }

    for (int i = 0; i < ctx->n_xlinks; ++i) {
        if (ctx->xlinks[i].seq > 0) {
            [ctx->xlinks[i].ev_done waitUntilSignaledValue:ctx->xlinks[i].seq timeoutMS:10000];
        }
        [ctx->xlinks[i].wrap_src release];
        [ctx->xlinks[i].wrap_dst release];
        [ctx->xlinks[i].peer_view release];
        [ctx->xlinks[i].peer_src release];
        [ctx->xlinks[i].ev_ready release];
        [ctx->xlinks[i].ev_done release];
        free(ctx->xlinks[i].host);
    }

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    ggml_metal_device_release_queue(ctx->dev, ctx->queue, ctx->owns_queue);

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }

    // in-flight staged uploads count as pending backend operations
    ggml_metal_device_upload_drain(ctx->dev);

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }

    for (int i = 0; i < ctx->prefetch_stage_count; ++i) {
        if (ctx->prefetch_stage_cmds[i]) {
            [ctx->prefetch_stage_cmds[i] release];
            ctx->prefetch_stage_cmds[i] = nil;
        }
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_read_batch_begin(ggml_metal_t ctx) {
    ggml_metal_device_read_batch_begin(ctx->dev);
}

void ggml_metal_read_batch_end(ggml_metal_t ctx) {
    ggml_metal_device_read_batch_end(ctx->dev);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        ggml_metal_prof_note(tensor->name, size, false);
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // Expert prefetch keeps the zero-copy source path while it fits the
        // device budget, then falls back to a bounded shared staging ring.
        if (ctx->is_prefetch) {
            int stage = -1;
            size_t offs_src = 0;
            id<MTLBuffer> buf_src = ggml_metal_device_wrap_host(ctx->dev, data, size, &offs_src);

            if (buf_src == nil) {
                stage = ctx->prefetch_stage_cur;
                ctx->prefetch_stage_cur = (ctx->prefetch_stage_cur + 1) % ctx->prefetch_stage_count;

                if (ctx->prefetch_stage_cmds[stage]) {
                    [ctx->prefetch_stage_cmds[stage] waitUntilCompleted];
                    [ctx->prefetch_stage_cmds[stage] release];
                    ctx->prefetch_stage_cmds[stage] = nil;
                }

                buf_src = ctx->prefetch_stages[stage];
                if (buf_src == nil || buf_src.length < size) {
                    [buf_src release];
                    buf_src = [device newBufferWithLength:size
                                                  options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
                    ctx->prefetch_stages[stage] = buf_src;
                }

                if (buf_src != nil) {
                    memcpy(buf_src.contents, data, size);
                    offs_src = 0;
                }
            }

            if (buf_src != nil) {
                id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
                id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

                [encoder copyFromBuffer:buf_src
                           sourceOffset:offs_src
                               toBuffer:bid_dst.metal
                      destinationOffset:bid_dst.offs
                                   size:size];

                [encoder endEncoding];
                [cmd_buf commit];

                [ctx->cmd_bufs_ext addObject:cmd_buf];
                ctx->cmd_buf_last = cmd_buf;
                if (stage >= 0) {
                    ctx->prefetch_stage_cmds[stage] = [cmd_buf retain];
                }

                [cmd_buf retain];

                return;
            }
        }

        // private buffers route small transfers through the device staging
        // buffer (synchronous) instead of allocating a Metal buffer per call
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!ctx->owns_queue && !ggml_metal_buffer_is_shared(buffer->context)
            && ggml_metal_device_stage_set(ctx->dev, bid_dst, data, size)) {
            return;
        }

        // wrap the source data into a Metal buffer
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ctx->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }

    @autoreleasepool {
        ggml_metal_prof_note(tensor->name, size, true);
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // private buffers route small transfers through the device staging
        // buffer (synchronous) instead of allocating a Metal buffer per call
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!ggml_metal_buffer_is_shared(buffer->context)
            && ggml_metal_device_stage_get(ctx->dev, bid_src, data, size)) {
            return;
        }

        id<MTLCommandQueue> queue = ctx->queue;

        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        if (buf_dst != nil) {
            // queue the copy operation into the queue of the Metal context
            // this will be queued at the end, after any currently ongoing GPU operations
            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
            [cmd_buf commit];
            [buf_dst release];

            // do not wait here for completion
            //[cmd_buf waitUntilCompleted];

            // instead, remember a reference to the command buffer and wait for it later if needed
            [ctx->cmd_bufs_ext addObject:cmd_buf];
            ctx->cmd_buf_last = cmd_buf;

            [cmd_buf retain];

            return;
        }

        // newBufferWithBytesNoCopy requires page-aligned data, and some drivers cap the size of
        // host-visible allocations; copy through a small staging buffer in chunks instead. The
        // staging path is synchronous (commit + waitUntilCompleted) so callers that would later
        // wait on cmd_buf_last still observe a completed transfer.
        size_t stage_size = MIN(size, (size_t) 8*1024*1024);
        id<MTLBuffer> buf_stage = nil;
        while (stage_size > 0) {
            buf_stage = [device newBufferWithLength:stage_size options:MTLResourceStorageModeShared];
            if (buf_stage != nil) {
                break;
            }
            stage_size /= 2;
        }
        if (buf_stage == nil) {
            GGML_LOG_ERROR("%s: failed to allocate staging buffer, size = %zu\n", __func__, size);
        }
        GGML_ASSERT(buf_stage);

        for (size_t done = 0; done < size;) {
            const size_t n = MIN(stage_size, size - done);

            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

            {
                id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

                [encoder copyFromBuffer:bid_src.metal
                           sourceOffset:bid_src.offs + done
                               toBuffer:buf_stage
                      destinationOffset:0
                                   size:n];

                [encoder endEncoding];
            }

            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];

            memcpy((char *) data + done, buf_stage.contents, n);

            done += n;
        }
    }
}

static struct ggml_metal_xdev_link * ggml_metal_xdev_link_get(
        ggml_metal_t ctx_src, ggml_metal_t ctx_dst) {
    struct ggml_metal_xdev_link * link = NULL;
    for (int i = 0; i < ctx_dst->n_xlinks; i++) {
        if (ctx_dst->xlinks[i].src_dev == ctx_src->dev) {
            link = &ctx_dst->xlinks[i];
            break;
        }
    }
    if (link == NULL) {
        if (ctx_dst->n_xlinks == 4) {
            return NULL;
        }
        link = &ctx_dst->xlinks[ctx_dst->n_xlinks++];
        link->src_dev = ctx_src->dev;
        id<MTLDevice> dev_src = ggml_metal_device_get_obj(ctx_src->dev);
        id<MTLDevice> dev_dst = ggml_metal_device_get_obj(ctx_dst->dev);
        link->ev_ready = [dev_src newSharedEvent];
        link->ev_done  = [dev_dst newSharedEvent];
        if (link->ev_ready == nil || link->ev_done == nil) {
            [link->ev_ready release];
            [link->ev_done release];
            memset(link, 0, sizeof(*link));
            ctx_dst->n_xlinks--;
            return NULL;
        }
    }

    return link;
}

// Direct copy between GPUs in the same Metal peer group.
static bool ggml_metal_cpy_xdev_peer(ggml_metal_t ctx_src, ggml_metal_t ctx_dst,
                                     const struct ggml_tensor * src, struct ggml_tensor * dst) {
#if TARGET_OS_OSX
    if (@available(macOS 10.15, *)) {
        id<MTLDevice> dev_src = ggml_metal_device_get_obj(ctx_src->dev);
        id<MTLDevice> dev_dst = ggml_metal_device_get_obj(ctx_dst->dev);
        const uint64_t peer_group = dev_src.peerGroupID;
        if (peer_group == 0 || peer_group != dev_dst.peerGroupID) {
            return false;
        }

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);
        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }
        id<MTLBuffer> metal_src = (id<MTLBuffer>) bid_src.metal;
        id<MTLBuffer> metal_dst = (id<MTLBuffer>) bid_dst.metal;

        const size_t size = ggml_nbytes(src);
        if (bid_src.offs > metal_src.length || size > metal_src.length - bid_src.offs ||
            bid_dst.offs > metal_dst.length || size > metal_dst.length - bid_dst.offs) {
            return false;
        }

        struct ggml_metal_xdev_link * link = ggml_metal_xdev_link_get(ctx_src, ctx_dst);
        if (link == NULL) {
            return false;
        }

        if (link->peer_src != metal_src) {
            // Wait before replacing a remote view still used by the last blit.
            if (link->peer_view != nil && link->seq > 0 &&
                ![link->ev_done waitUntilSignaledValue:link->seq timeoutMS:10000]) {
                return false;
            }

            [link->peer_view release];
            [link->peer_src release];
            link->peer_view = [metal_src newRemoteBufferViewForDevice:dev_dst];
            link->peer_src  = link->peer_view != nil ? [metal_src retain] : nil;
            if (link->peer_view == nil) {
                return false;
            }
        }

        const uint64_t seq = ++link->seq;
        @autoreleasepool {
            id<MTLCommandBuffer> cmd_src = [ctx_src->queue commandBuffer];
            [cmd_src encodeSignalEvent:link->ev_ready value:seq];
            [cmd_src commit];
            [ctx_src->cmd_bufs_ext addObject:cmd_src];
            ctx_src->cmd_buf_last = cmd_src;
            [cmd_src retain];

            id<MTLCommandBuffer> cmd_dst = [ctx_dst->queue commandBuffer];
            [cmd_dst encodeWaitForEvent:link->ev_ready value:seq];
            id<MTLBlitCommandEncoder> encoder = [cmd_dst blitCommandEncoder];
            [encoder copyFromBuffer:link->peer_view sourceOffset:bid_src.offs
                           toBuffer:metal_dst destinationOffset:bid_dst.offs size:size];
            [encoder endEncoding];
            [cmd_dst encodeSignalEvent:link->ev_done value:seq];
            [cmd_dst commit];
            [ctx_dst->cmd_bufs_ext addObject:cmd_dst];
            ctx_dst->cmd_buf_last = cmd_dst;
            [cmd_dst retain];
        }

        if (!link->peer_logged) {
            // fprintf, not GGML_LOG_INFO, so the confirmation shows at the default
            // log level like the SIMD-width line. Deduped by peer group across contexts:
            // a tool that reloads the model per run rebuilds the link every time.
            static uint64_t logged[8];
            static int      n_logged = 0;

            bool seen = false;
            for (int i = 0; i < n_logged; ++i) {
                seen = seen || logged[i] == peer_group;
            }

            if (!seen) {
                fprintf(stderr, "ggml_metal: Infinity Fabric peer transfer enabled: %s -> %s (peer group %llu)\n",
                        dev_src.name.UTF8String, dev_dst.name.UTF8String,
                        (unsigned long long) peer_group);
                if (n_logged < (int) (sizeof(logged)/sizeof(logged[0]))) {
                    logged[n_logged++] = peer_group;
                }
            }

            link->peer_logged = true;
        }
        return true;
    }
#else
    GGML_UNUSED(ctx_src);
    GGML_UNUSED(ctx_dst);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
#endif
    return false;
}

// event-chained hand-off: both blits share one host block and order through
// MTLSharedEvents, so neither GPU drains and the CPU never blocks
static bool ggml_metal_cpy_xdev_events(ggml_metal_t ctx_src, ggml_metal_t ctx_dst,
                                       const struct ggml_tensor * src, struct ggml_tensor * dst) {
    struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
    struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);
    if (bid_src.metal == nil || bid_dst.metal == nil) {
        return false;
    }

    const size_t size = ggml_nbytes(src);

    struct ggml_metal_xdev_link * link = ggml_metal_xdev_link_get(ctx_src, ctx_dst);
    if (link == NULL) {
        return false;
    }
    if (link->cap == 0) {
        GGML_LOG_INFO("%s: cross-device hand-off via shared events and host staging (TOSH_MGPU_EVENTS)\n", __func__);
    }

    if (link->cap < size) {
        // resizing frees pages a queued blit may still read; wait out the link first
        if (link->seq > 0 && ![link->ev_done waitUntilSignaledValue:link->seq timeoutMS:10000]) {
            return false;
        }
        [link->wrap_src release];
        [link->wrap_dst release];
        free(link->host);
        link->cap = MAX(size, (size_t) 1*1024*1024);
        if (posix_memalign(&link->host, (size_t) sysconf(_SC_PAGESIZE), link->cap) != 0) {
            link->host = NULL;
            link->cap  = 0;
            return false;
        }
        id<MTLDevice> dev_src = ggml_metal_device_get_obj(ctx_src->dev);
        id<MTLDevice> dev_dst = ggml_metal_device_get_obj(ctx_dst->dev);
        link->wrap_src = [dev_src newBufferWithBytesNoCopy:link->host
                              length:link->cap options:MTLResourceStorageModeShared deallocator:nil];
        link->wrap_dst = [dev_dst newBufferWithBytesNoCopy:link->host
                              length:link->cap options:MTLResourceStorageModeShared deallocator:nil];
        if (link->wrap_src == nil || link->wrap_dst == nil) {
            return false;
        }
    }

    const uint64_t seq = ++link->seq;

    @autoreleasepool {
        // reuse the host block only after the destination consumed the last hand-off
        id<MTLCommandBuffer> cmd_src = [ctx_src->queue commandBuffer];
        if (seq > 1) {
            [cmd_src encodeWaitForEvent:link->ev_done value:seq - 1];
        }
        id<MTLBlitCommandEncoder> enc_src = [cmd_src blitCommandEncoder];
        [enc_src copyFromBuffer:bid_src.metal sourceOffset:bid_src.offs
                       toBuffer:link->wrap_src destinationOffset:0 size:size];
        [enc_src endEncoding];
        [cmd_src encodeSignalEvent:link->ev_ready value:seq];
        [cmd_src commit];
        [ctx_src->cmd_bufs_ext addObject:cmd_src];
        ctx_src->cmd_buf_last = cmd_src;
        [cmd_src retain];

        id<MTLCommandBuffer> cmd_dst = [ctx_dst->queue commandBuffer];
        [cmd_dst encodeWaitForEvent:link->ev_ready value:seq];
        id<MTLBlitCommandEncoder> enc_dst = [cmd_dst blitCommandEncoder];
        [enc_dst copyFromBuffer:link->wrap_dst sourceOffset:0
                       toBuffer:bid_dst.metal destinationOffset:bid_dst.offs size:size];
        [enc_dst endEncoding];
        [cmd_dst encodeSignalEvent:link->ev_done value:seq];
        [cmd_dst commit];
        [ctx_dst->cmd_bufs_ext addObject:cmd_dst];
        ctx_dst->cmd_buf_last = cmd_dst;
        [cmd_dst retain];
    }

    return true;
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    // A Metal blit can't reach a buffer owned by another device, so a multi-GPU layer
    // hand-off goes through host memory (GGML_METAL_CROSS_STAGING_DISABLE uses the generic copy).
    if (ctx_src->dev != ctx_dst->dev) {
        if (getenv("GGML_METAL_CROSS_STAGING_DISABLE") != NULL) {
            return false;
        }

        // both buffers must be private for the cross-device blits
        static int use_events = -1;
        if (use_events < 0) {
            const char * v = getenv("TOSH_MGPU_EVENTS");
            use_events = (v && v[0] == '1') ? 1 : 0;
        }
        // Infinity Fabric peer copy is opt-in until validated on bridged hardware:
        // a success that silently returns wrong data has no fallback.
        static int use_peer = -1;
        if (use_peer < 0) {
            const char * v = getenv("TOSH_MGPU_PEER");
            use_peer = (v && v[0] == '1') ? 1 : 0;
        }
        ggml_backend_buffer_t ebsrc = src->view_src ? src->view_src->buffer : src->buffer;
        ggml_backend_buffer_t ebdst = dst->view_src ? dst->view_src->buffer : dst->buffer;
        if (!ggml_metal_buffer_is_shared(ebsrc->context) &&
            !ggml_metal_buffer_is_shared(ebdst->context)) {
            if (use_peer && ggml_metal_cpy_xdev_peer(ctx_src, ctx_dst, src, dst)) {
                return true;
            }
            if (use_events && ggml_metal_cpy_xdev_events(ctx_src, ctx_dst, src, dst)) {
                return true;
            }
        }
        // drain both GPUs: the source before reading it, the destination before
        // overwriting an input its in-flight graph may still be consuming
        ggml_metal_synchronize(ctx_src);
        ggml_metal_synchronize(ctx_dst);

        ggml_backend_buffer_t bsrc = src->view_src ? src->view_src->buffer : src->buffer;
        ggml_backend_buffer_t bdst = dst->view_src ? dst->view_src->buffer : dst->buffer;
        ggml_metal_buffer_t buf_src = (ggml_metal_buffer_t) bsrc->context;
        ggml_metal_buffer_t buf_dst = (ggml_metal_buffer_t) bdst->context;

        const size_t size = ggml_nbytes(src);
        void * stage = malloc(size);
        if (!stage) {
            return false;
        }
        ggml_metal_buffer_get_tensor(buf_src, src, stage, 0, size);
        ggml_metal_buffer_set_tensor(buf_dst, dst, stage, 0, size);
        free(stage);
        return true;
    }

    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ctx_src->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            ctx->capture_compute = -1;

            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ctx->queue;

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ctx->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ctx->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            idx_start,
            idx_end,
            ctx->use_fusion,
            ctx->use_concurrency,
            ctx->capture_compute,
            ctx->debug_graph,
            ctx->debug_fusion);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
