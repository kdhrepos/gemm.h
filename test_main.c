#include "gemm.h"
#include "test.h"

static D_TYPE parse_dtype(char* dtype);
static char* to_lowercase(char* str);
static void is_valid_integer(const char* input, const char* arg_type);
static void help();

static D_TYPE parse_dtype(char* dtype) {
  dtype = to_lowercase(dtype);

  if (!strcmp(dtype, "all") || !strcmp(dtype, "a"))
    return D_ALL;
  else if (!strcmp(dtype, "float") || !strcmp(dtype, "s") ||
           !strcmp(dtype, "f"))
    return D_FP32;
  else if (!strcmp(dtype, "double") || !strcmp(dtype, "d"))
    return D_FP64;
  else if (!strcmp(dtype, "int32") || !strcmp(dtype, "i"))
    return D_INT32;
  else if (!strcmp(dtype, "hfloat") || !strcmp(dtype, "h")) {
    fprintf(stderr, "Unsupported datatype. Use --help for usage.\n");
    exit(EXIT_FAILURE);
  } else if (!strcmp(dtype, "bfloat") || !strcmp(dtype, "bf")) {
    fprintf(stderr, "Unsupported datatype. Use --help for usage.\n");
    exit(EXIT_FAILURE);
  } else if (!strcmp(dtype, "int16") || !strcmp(dtype, "hq")) {
    return D_INT16;
  } else if (!strcmp(dtype, "int8") || !strcmp(dtype, "q")) {
    return D_INT8;
  } else {
    fprintf(stderr, "Unknown datatype. Use --help for usage.\n");
    exit(EXIT_FAILURE);
  }
}

static char* to_lowercase(char* str) {
  for (int i = 0; str[i] != '\0'; i++) {
    str[i] = tolower((unsigned char)str[i]);
  }
  return str;
}

static void is_valid_integer(const char* input, const char* arg_type) {
  if (optarg == NULL || (*input) == '\0') {
    fprintf(stderr, "[Error]: Missing argument for --iter or -i.\n");
    fprintf(stderr, "Use --help for usage");
    exit(EXIT_FAILURE);
  }
  char* p = (char*)input;
  while (*p) {
    if (!isdigit(*p)) {
      fprintf(stderr, "[Error]: Invalid integer for %s.\n", arg_type);
      fprintf(stderr, "Use --help for usage");
      exit(EXIT_FAILURE);
    }
    p++;
  }
}

static void help() {
  fprintf(stderr, "GEMM.H Tester %s\n", GEMM_H_TESTER_VERSION);
  fprintf(stderr, "  -h, --help             Print this help message\n");
  fprintf(stderr, "  -t, --type=<dtype>     Data type \n");
  fprintf(stderr, "                         s:  float \n");
  fprintf(stderr, "                         d:  double\n");
  fprintf(stderr, "                         i:  int32 \n");
  fprintf(stderr, "                         q:  int16 \n");
  fprintf(stderr, "                         hq: int8  \n");
  fprintf(stderr,
          "                         h:  hfloat "
          "[Unsupported]\n");
  fprintf(stderr,
          "                         bf: bfloat "
          "[Unsupported]\n");
  fprintf(stderr,
          "  -i, --iter=<num>       Number of iteration for each M, K, N \n");
  fprintf(stderr,
          "  -m, --M=<size>         Matrix size M for C(MxN) = A(MxK) X B(KxN) "
          "Default: 1024\n");
  fprintf(stderr,
          "  -k, --K=<size>         Matrix size K for C(MxN) = A(MxK) X B(KxN) "
          "Default: 1024\n");
  fprintf(stderr,
          "  -n, --N=<size>         Matrix size N for C(MxN) = A(MxK) X B(KxN) "
          "Default: 1024\n");
  fprintf(stderr,
          "  -r, --range=<num>      Iterate GEMM over three for loops\n");
  fprintf(stderr, "                         for M to (M + range)\n");
  fprintf(stderr, "                             for N to (N + range)\n");
  fprintf(stderr, "                                 for K to (K + range)\n");
  fprintf(
      stderr,
      "  -b, --bound=<num>      Bound for generating random matrix value \n");
  fprintf(stderr,
          "  -f, --file=<filename>  Print the GEMM output to <filename>\n");
  fprintf(stderr,
          "  -p, --print            Print the GEMM output to console \n");
  fprintf(stderr,
          "  -c, --compare          Also benchmark naive GEMM and report "
          "speedup \n");
}

int main(int argc, char* argv[]) {
  int opt;
  int M = 1024, N = 1024, K = 1024;
  int niter = 1, range = 1, bound = 5;
  BOOL console_flag = FALSE;
  BOOL compare_flag = FALSE;
  FILE* file = NULL;
  D_TYPE dtype = D_FP32;

  static struct option long_options[] = {{"type", required_argument, 0, 't'},
                                         {"M", required_argument, 0, 'm'},
                                         {"K", required_argument, 0, 'k'},
                                         {"N", required_argument, 0, 'n'},
                                         {"iter", required_argument, 0, 'i'},
                                         {"range", required_argument, 0, 'r'},
                                         {"bound", required_argument, 0, 'b'},
                                         {"file", required_argument, 0, 'f'},
                                         {"print", no_argument, 0, 'p'},
                                         {"compare", no_argument, 0, 'c'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  while ((opt = getopt_long(argc, argv, "t:m:k:n:i:r:b:f:pch", long_options,
                            NULL)) != -1) {
    switch (opt) {
      case 't':
        dtype = parse_dtype(optarg);
        break;
      case 'i':
        is_valid_integer(optarg, "--iter");
        niter = atoi(optarg);
        break;
      case 'm':
        is_valid_integer(optarg, "--M");
        M = atoi(optarg);
        break;
      case 'n':
        is_valid_integer(optarg, "--N");
        N = atoi(optarg);
        break;
      case 'k':
        is_valid_integer(optarg, "--K");
        K = atoi(optarg);
        break;
      case 'b':
        is_valid_integer(optarg, "--bound");
        bound = atoi(optarg);
        break;
      case 'r':
        is_valid_integer(optarg, "--range");
        range = atoi(optarg);
        break;
      case 'p':
        console_flag = TRUE;
        break;
      case 'c':
        compare_flag = TRUE;
        break;
      case 'f':
        if ((file = fopen(optarg, "a")) == NULL) {
          perror("[Error]: File open failed\n");
          exit(EXIT_FAILURE);
        }
        break;
      case 'h':
        help();
        exit(EXIT_SUCCESS);
        break;
      case '?':
        fprintf(stderr, "[Error]: Unknown option.\n");
        fprintf(stderr, "Use --help for usage.\n");
        exit(EXIT_FAILURE);
        // default:
        //     help(argv[0]);
        //     exit(EXIT_SUCCESS);
    }
  }

  switch (dtype) {
    case D_ALL: {
      sgemm_test(M, N, K, niter, range, bound, file, console_flag,
                 compare_flag);
      dgemm_test(M, N, K, niter, range, bound, file, console_flag,
                 compare_flag);
      igemm_test(M, N, K, niter, range, bound, file, console_flag,
                 compare_flag);
      hqgemm_test(M, N, K, niter, range, bound, file, console_flag,
                  compare_flag);
      qgemm_test(M, N, K, niter, range, bound, file, console_flag,
                 compare_flag);
      break;
    }
    case D_FP32: {
      sgemm_test(M, N, K, niter, range, bound, file, console_flag,
                 compare_flag);
      break;
    }
    case D_FP64: {
      dgemm_test(M, N, K, niter, range, bound, file, console_flag,
                 compare_flag);
      break;
    }
    case D_INT32: {
      igemm_test(M, N, K, niter, range, bound, file, console_flag,
                 compare_flag);
      break;
    }
    case D_INT16: {
      hqgemm_test(M, N, K, niter, range, bound, file, console_flag,
                  compare_flag);
      break;
    }
    case D_INT8: {
      qgemm_test(M, N, K, niter, range, bound, file, console_flag,
                 compare_flag);
      break;
    }
    default: {
      fprintf(stderr, "[Error]: Unknown datatype.\n");
      fprintf(stderr, "Use --help for usage.\n");
      exit(EXIT_FAILURE);
    }
  }

  if (file != NULL)
    fclose(file);

  return 0;
}