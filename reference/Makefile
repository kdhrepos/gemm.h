all:
	gcc -O2 -march=native -std=c11 -I.. openblas_ref.c ../report.c -o openblas_ref \
	    -L$(HOME)/opt/openblas/lib -Wl,-rpath,$(HOME)/opt/openblas/lib -lopenblas
