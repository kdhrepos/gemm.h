/**********************************************************************************************
 * File   : report.c
 * Author : kdh
 * Github : https://github.com/kdhrepos/gemm.h
 *
 * Description:
 *      Printers for TestResult. Both formats are shared by gemm.h's tester (run)
 *      and reference/openblas_ref.
 *
**********************************************************************************************/

#include "report.h"

/* Columns are the same for every library, so the rows can be compared directly. */
static void print_csv(FILE* stream, const TestResult* res) {
  static int header_done[2] = {0, 0}; /* stdout, file */
  int* done = &header_done[(stream == stdout) ? 0 : 1];

  if (!*done) {
    fprintf(stream,
            "lib,threads,M,N,K,iters,avg_ms,min_ms,avg_gflops,max_gflops\n");
    *done = 1;
  }
  fprintf(stream, "%s:%s,%d,%d,%d,%d,%d,%.4f,%.4f,%.1f,%.1f\n", res->lib,
          res->name, res->threads, res->M, res->N, res->K, res->niter,
          res->exec_time[0] / res->niter, res->exec_time[2],
          res->gflops[0] / res->niter, res->gflops[1]);
}

static void print_text(FILE* stream, const TestResult* res) {
  const double avg_ms = res->exec_time[0] / res->niter;
  const double avg_gflops = res->gflops[0] / res->niter;

  fprintf(stream, "════════════════════════════════════════════════\n");
  fprintf(stream, " %s:%s   A: %dx%d   B: %dx%d   C: %dx%d\n", res->lib,
          res->name, res->M, res->K, res->K, res->N, res->M, res->N);
  fprintf(stream, " threads: %d   warm-up: %d   iters: %d   ISA: %s\n",
          res->threads, res->warmup, res->niter, res->isa);
  fprintf(stream, "────────────────────────────────────────────────\n");
  fprintf(stream, " Result   : %s\n\n",
          res->is_valid ? "[Valid GEMM]" : "[Invalid GEMM!]");

  fprintf(stream, "%12s%9s%9s%9s\n", "", "best", "avg", "worst");
  fprintf(stream, " Time(ms) : %9.3f%9.3f%9.3f\n", res->exec_time[2], avg_ms,
          res->exec_time[1]);
  fprintf(stream, " GFLOPS   : %9.3f%9.3f%9.3f\n", res->gflops[1], avg_gflops,
          res->gflops[2]);

  if (res->compare) {
    fprintf(stream,
            "\n vs Naive : %.3fms (%.3f GFLOPS) -> %.3fx speedup (avg)\n",
            res->naive_time, res->naive_gflops,
            (avg_ms > 0) ? (res->naive_time / avg_ms) : 0);
  }
  fprintf(stream, "════════════════════════════════════════════════\n");
}

void print_result(FILE* stream, const TestResult* res, const int csv) {
  if (csv)
    print_csv(stream, res);
  else
    print_text(stream, res);
}
