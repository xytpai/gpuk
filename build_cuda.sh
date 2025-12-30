ARCH=${ARCH:-sm_90}
nvcc -x cu -O3 ${@:2} --std=c++20 --expt-relaxed-constexpr -arch ${ARCH} -Isrc/common -Isrc/impl $1 -o a.out
