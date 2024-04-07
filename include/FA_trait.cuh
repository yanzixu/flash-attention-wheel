#ifndef _FA_TRAIT_
#define _FA_TRAIT_

namespace FA_FWD_CONFIG
{
    using namespace cute;

    template <typename T_, int TileM_, int TileN_, int HeadDim_, int Warps>
    struct yan_trait
    {
        static constexpr int kTileM = TileM_;
        static constexpr int kTileN = TileN_;
        static constexpr int kHeadDim = HeadDim_;
        static constexpr int kWarps = Warps;
        static constexpr int kBlockSize = Warps * 32;
        using T = T_;
        using AccType = float;
        // TODO: g2s cp
        static constexpr int kElements_128bit = sizeof(uint128_t) / sizeof(T);
        static constexpr int kThreadsPerRow = kHeadDim / kElements_128bit;
        static constexpr int kTHreadsRows = kBlockSize / kThreadsPerRow;
        using G2S_OP = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
        using G2S_ATOM = Copy_Atom<G2S_OP, T>;
        using G2S_COPY = decltype(make_tiled_copy(G2S_ATOM{},
                                                  Layout<Shape<Int<kTHreadsRows>, Int<kThreadsPerRow>>, Stride<Int<kThreadsPerRow>, _1>>{},
                                                  Layout<Shape<_1, Int<kElements_128bit>>>{}));
        // TODO:shared memory layout
        static constexpr int kLD_SWZ_M = 3;
        static constexpr int kLD_SWZ_B = 3;
        static constexpr int kLD_SWZ_S = 4;
        using SMEM_ATOM = decltype(composition(Swizzle<kLD_SWZ_B, kLD_SWZ_M, kLD_SWZ_S>{},
                                               Layout<Shape<Int<kTHreadsRows>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>{}));
        using SMEM_ATOM_TRANS = decltype(composition(Swizzle<kLD_SWZ_B, kLD_SWZ_M, kLD_SWZ_S>{},
                                                     Layout<Shape<Int<kHeadDim>, Int<kTHreadsRows>>, Stride<_1, Int<kHeadDim>>>{}));
        using SMEM_LAYOUT_Q = decltype(tile_to_shape(SMEM_ATOM{}, Shape<Int<kTileM>, Int<kHeadDim>>{}));
        using SMEM_LAYOUT_K = decltype(tile_to_shape(SMEM_ATOM{}, Shape<Int<kTileN>, Int<kHeadDim>>{}));
        // TODO: yzx design sV layout failed
        using SMEM_LAYOUT_V = SMEM_LAYOUT_K;
        // g2s 过程直接将sV构造成列主序的layout在无法编译通过，C:\code\cutlass\include\cute/atom/copy_traits.hpp(123): error : static assertion failed with "In CopyAtom, dst layout doesn't vectorize into registers. This dst layout is incompatible with this tiled copy."
        using SMEM_LAYOUT_VT = decltype(tile_to_shape(SMEM_ATOM_TRANS{}, Shape<Int<kHeadDim>, Int<kTileN>>{}));

        // TODO: FA sV Layout 和yzx 设计的加载到寄存器的数据没区别
        // using SmemLayoutAtomV = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, Int<64>>, Stride<Int<64>, _1>>{}));
        // using SMEM_LAYOUT_V = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<kTileN>, Int<kHeadDim>>{}));
        // using SmemLayoutAtomVtransposedNoSwizzle = Layout<Shape<Int<64>, Int<kTileN>>, Stride<_1, Int<64>>>;
        // using SmemLayoutAtomVtransposed = decltype(composition(Swizzle<3, 3, 3>{}, SmemLayoutAtomVtransposedNoSwizzle{}));
        // using SMEM_LAYOUT_VT = decltype(tile_to_shape(SmemLayoutAtomVtransposed{}, Shape<Int<kHeadDim>, Int<kTileN>>{}));
        // static constexpr int kThreadsPerRow_V = 64 / kElements_128bit;
        // static constexpr int kTHreadsRows_V = kBlockSize / kThreadsPerRow_V;
        // using G2S_COPY_V = decltype(make_tiled_copy(G2S_ATOM{},
        //                                             Layout<Shape<Int<kTHreadsRows_V>, Int<kThreadsPerRow_V>>, Stride<Int<kThreadsPerRow_V>, _1>>{},
        //                                             Layout<Shape<_1, Int<kElements_128bit>>>{}));

        // TODO: VT tile test
        using SMEM_ATOM_Test = decltype(composition(Swizzle<kLD_SWZ_B, kLD_SWZ_M, kLD_SWZ_S>{},
                                                    Layout<Shape<Int<16>, Int<kTileN>>, Stride<Int<kTileN>, _1>>{}));
        using SMEM_LAYOUT_VTest = decltype(tile_to_shape(SMEM_ATOM_Test{}, Shape<Int<kHeadDim>, Int<kTileN>>{}));
        static constexpr int SMEM_SIZE_Q = cosize(SMEM_LAYOUT_Q{});
        static constexpr int SMEM_SIZE_K = cosize(SMEM_LAYOUT_K{});
        static constexpr int SMEM_SIZE_V = SMEM_SIZE_K;
        // TODO::tiled mma
        using MMA_OP_FIRST = SM80_16x8x16_F32F16F16F32_TN;
        using MMA_ATOM_FIRST = MMA_Atom<MMA_OP_FIRST>;
        using TILED_MMA_FIRST = decltype(make_tiled_mma(MMA_ATOM_FIRST{}, Layout<Shape<Int<kWarps>, _1, _1>>{}, Layout<Shape<_1, _2, _1>>{}));

        // using MMA_OP_SECOND = SM80_16x8x16_F16F16F16F16_TN;
        using MMA_OP_SECOND = MMA_OP_FIRST;
        using MMA_ATOM_SECOND = MMA_Atom<MMA_OP_SECOND>;
        using TILED_MMA_SECOND = decltype(make_tiled_mma(MMA_ATOM_SECOND{}, Layout<Shape<Int<kWarps>, _1, _1>>{}, Layout<Shape<_1, _2, _1>>{}));

        // TODO: s2r cp
        using S2R_ATOM = Copy_Atom<SM75_U32x4_LDSM_N, T>;
        using S2R_ATOM_V = Copy_Atom<SM75_U16x8_LDSM_T, T>;

        // TODO: r2s cp
        using SMEM_LAYOUT_O = SMEM_LAYOUT_Q;
        using R2S_ATOM = Copy_Atom<UniversalCopy<float>, T>;
        using R2S_ATOM_TEST = Copy_Atom<DefaultCopy, T>;
        using S2G_ATOM = Copy_Atom<DefaultCopy, T>;
        // TODO: s2g cp
        using S2G_COPY = decltype(make_tiled_copy(S2G_ATOM{},
                                                  Layout<Shape<Int<kTHreadsRows>, Int<kThreadsPerRow>>, Stride<Int<kThreadsPerRow>, _1>>{},
                                                  Layout<Shape<_1, Int<kElements_128bit>>>{}));
    };
}

#endif