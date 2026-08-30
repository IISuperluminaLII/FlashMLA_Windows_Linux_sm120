#pragma once
// Ints per tile-scheduler metadata row, produced by csrc/smxx/get_mla_metadata.cu:
// [begin_idx, begin_block_idx, end_idx, end_block_idx(exclusive), begin_n_split_idx, _, _, _].
// ONE definition shared by the dense and sparse decode params headers (defining it in
// each would be a same-TU redefinition in pybind.cpp, which includes both). MUST equal
// the top-level ::TileSchedulerMetaDataSize (csrc/params.h) -- pinned by a static_assert
// in pybind.cpp, the one TU that sees both constants, so drift cannot compile.

namespace sm120 {

static constexpr int TileSchedulerMetaDataSize = 8;

}  // namespace sm120
