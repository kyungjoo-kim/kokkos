/*
//@HEADER
// ************************************************************************
// 
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
// 
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
// 
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// When multiple device spaces are activated, we still need to identify 
// which execution space is used for testing. Some experimental features
// implemented on a certain device only.

#if defined( KOKKOS_HAVE_OPENMP )

typedef Kokkos::OpenMP TestHostDevice ;
const char TestHostDeviceName[] = "Kokkos::OpenMP" ;
#define PERF_TEST_SELECT_OPENMP

#elif defined( KOKKOS_HAVE_PTHREAD )

typedef Kokkos::Threads TestHostDevice ;
const char TestHostDeviceName[] = "Kokkos::Threads" ;
#define PERF_TEST_SELECT_THREADS

#elif defined( KOKKOS_HAVE_SERIAL )

typedef Kokkos::Serial TestHostDevice ;
const char TestHostDeviceName[] = "Kokkos::Serial" ;
#define PERF_TEST_SELECT_SERIAL

#else
#  error "You must enable at least one of the following execution spaces in order to build this test: Kokkos::Threads, Kokkos::OpenMP, or Kokkos::Serial."
#endif

#include <impl/Kokkos_Timer.hpp>

#include <PerfTestHexGrad.hpp>
#include <PerfTestBlasKernels.hpp>
#include <PerfTestGramSchmidt.hpp>

#if defined( PERF_TEST_SELECT_THREADS ) || defined( PERF_TEST_SELECT_SERIAL )
#include <Kokkos_Threads.hpp>
#include <Threads/Kokkos_Threads_TaskPolicy.hpp>
#include <impl/Kokkos_Serial_TaskPolicy.hpp>
#include <PerfTestTeamBarrier.hpp>
#endif

#include <PerfTestDriver.hpp>

//------------------------------------------------------------------------

namespace Test {

class host : public ::testing::Test {
protected:
  static void SetUpTestCase()
  {
    const unsigned numa_count = Kokkos::hwloc::get_available_numa_count();
    const unsigned cores_per_numa = Kokkos::hwloc::get_available_cores_per_numa();
    const unsigned threads_per_core = Kokkos::hwloc::get_available_threads_per_core();
    const unsigned threads_count = numa_count * cores_per_numa * threads_per_core;

    TestHostDevice::initialize( threads_count, numa_count, cores_per_numa );
    TestHostDevice::print_configuration( std::cout, true );
  }

  static void TearDownTestCase()
  {
    TestHostDevice::finalize();
  }
};

TEST_F( host, hexgrad ) {
  EXPECT_NO_THROW(run_test_hexgrad< TestHostDevice>( 10, 20, TestHostDeviceName ));
}

TEST_F( host, gramschmidt ) {
  EXPECT_NO_THROW(run_test_gramschmidt< TestHostDevice>( 10, 20, TestHostDeviceName ));
}

#if defined( PERF_TEST_SELECT_THREADS ) || defined( PERF_TEST_SELECT_SERIAL )
TEST_F( host, teambarrier ) {
  EXPECT_NO_THROW(run_test_teambarrier< TestHostDevice>( 10, 20, 10, TestHostDeviceName ));
}
#endif

} // namespace Test


