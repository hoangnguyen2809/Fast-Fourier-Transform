CXX = g++
MPICXX = mpic++
CXXFLAGS = -std=c++14 -O3 $(MACRO)
LDFLAGS = -lpthread

COMMON= core/utils.h core/cxxopts.h core/get_time.h 
SERIAL= fast_fourier_transform_serial
PARALLEL= fast_fourier_transform_parallel 
DISTRIBUTED= fast_fourier_transform_distributed
ALL= $(SERIAL) $(PARALLEL) $(DISTRIBUTED)

all : $(ALL)

$(SERIAL): %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(PARALLEL): %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(DISTRIBUTED): %: %.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

.PHONY : clean

clean :
	rm -f *.o *.obj $(ALL)
