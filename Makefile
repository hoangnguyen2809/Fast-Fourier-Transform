CXX = g++
MPICXX = mpic++
CXXFLAGS = -std=c++14 -O3 $(MACRO)

COMMON= core/utils.h core/cxxopts.h core/get_time.h 
SERIAL= longest_common_sequence_serial
PARALLEL= longest_common_sequence_parallel 
DISTRIBUTED= longest_common_sequence_distributed
ALL= $(SERIAL) $(PARALLEL)

all : $(ALL)

$(SERIAL): %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(PARALLEL): %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(DISTRIBUTED): %: %.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

.PHONY : clean

clean :
	rm -f *.o *.obj $(ALL)
