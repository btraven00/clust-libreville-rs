// brrr-gve's gve/leiden.cxx wrapper (MIT; GVE-Leiden, Sahu, upstream f741a40, vendored unchanged in
// $PHD/brrr-gve/gve/inc), plus a clock around leidenStaticOmp alone: the "algo only" time, which
// excludes the DiGraph build and the relabelling, as clust-libreville-rs's timing excludes Graph::from_csr.
// Built by bench/py/run_member.py with -I $PHD/brrr-gve/gve.
#include <chrono>
#include <cstdint>
#include "inc/main.hxx"

static double last_algo_seconds = 0;

extern "C" double gve_last_algo_seconds() { return last_algo_seconds; }

extern "C" int gve_leiden(int64_t n, const int64_t *indptr, const int32_t *indices, const float *weights,
                          double resolution, const int32_t *perm, int32_t *labels) {
  using K = uint32_t;
  DiGraph<K, None, float> x;
  addVerticesU(x, K(0), K(n));
  #pragma omp parallel
  {
    for (int64_t u = 0; u < n; ++u) {
      if (!belongsOmp(K(perm[u]))) continue;  // each thread fills only the rows it owns
      for (int64_t j = indptr[u]; j < indptr[u + 1]; ++j)
        x.addEdge(K(perm[u]), K(perm[indices[j]]), weights[j]);
    }
  }
  updateOmpU(x);
  auto t = std::chrono::steady_clock::now();
  auto a = leidenStaticOmp(x, {1, resolution});
  last_algo_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  for (int64_t u = 0; u < n; ++u) labels[u] = int32_t(a.membership[perm[u]]);
  return a.passes;
}
