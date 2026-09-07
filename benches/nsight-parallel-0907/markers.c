#include <nvtx3/nvToolsExt.h>
void mark_push(const char *name) { nvtxRangePushA(name); }
void mark_pop(void) { nvtxRangePop(); }
