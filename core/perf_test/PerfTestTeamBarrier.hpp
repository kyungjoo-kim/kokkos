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

#include <cmath>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Test {
  
  template<typename ScalarType, class DeviceType, bool ApplyTeamBarrier>
  class TaskTeamFunctor {
  public:
    typedef int value_type ;
    typedef DeviceType execution_space ;
    typedef Kokkos::Experimental::TaskPolicy<execution_space>  policy_type ;
    typedef ScalarType scalar_type;

    typedef typename policy_type::member_type member_type;
    typedef typename execution_space::size_type size_type ;

    typedef Kokkos::View<scalar_type*,execution_space,Kokkos::MemoryUnmanaged> scalar_type_array;

    const size_type itask;
    const scalar_type_array a, b;

    TaskTeamFunctor(const size_type & arg_itask, 
                    const scalar_type_array & arg_a,
                    const scalar_type_array & arg_b) 
      : itask(arg_itask), a(arg_a), b(arg_b) {}
    
    TaskTeamFunctor(const TaskTeamFunctor & rhs)
      : itask(rhs.itask), a(rhs.a), b(rhs.b) {}
    
    enum { Big   = 1000,
           Small = 100};

    KOKKOS_INLINE_FUNCTION
    void apply(value_type & r_val) {
      r_val = 0;
    }
    
    KOKKOS_INLINE_FUNCTION
    void apply(const member_type & member, value_type & r_val) {
      const auto range = a.dimension(0);
      for (auto iter=0;iter<Big;++iter) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, range),
                             [&](const int i) {
                               b(i) = itask*range + i;
                               scalar_type tmp = range*itask;
                               for (auto j=0;j<Small;++j)
                                 tmp += j;
                               a(i) = b(i) + tmp;
                             });
        if (ApplyTeamBarrier)
          member.team_barrier();
      }
    }
  };

  template<typename ScalarType, class DeviceType, bool ApplyTeamBarrier = true>
  struct TeamBarrier {
    typedef DeviceType execution_space;
    typedef ScalarType scalar_type;
    typedef typename execution_space::size_type size_type ;

    static double test(const size_t nwork,
                       const size_t nwork_per_task,
                       const size_t team_size,
                       const size_t iter = 1) {
      typedef Kokkos::Experimental::TaskPolicy<execution_space> policy_type;
      typedef Kokkos::Experimental::Future<int,execution_space> future_type;

      typedef TaskTeamFunctor<scalar_type,execution_space,ApplyTeamBarrier> functor_type;
      typedef Kokkos::View<scalar_type*,execution_space> scalar_type_array;

      scalar_type_array a("a", nwork), b("b", nwork);

      const size_t ntasks = (nwork/nwork_per_task + 1);
      policy_type policy(ntasks*2,
                         (sizeof(functor_type) + 128)/128*128,
                         0, team_size);
      
      double dt_min = 0;

      Kokkos::Impl::Timer timer;

      for (int i=0;i<iter;++i) {
        timer.reset();
        for (int itask=0;i<ntasks;++itask) {
          const int begin = itask*nwork_per_task, tmp = begin + nwork_per_task; 
          const int end   = (tmp < nwork ? tmp : nwork);

          auto aa = Kokkos::subview(a, Kokkos::pair<int,int>(begin,end));
          auto bb = Kokkos::subview(b, Kokkos::pair<int,int>(begin,end));
          
          future_type f = policy.proc_create_team(functor_type(itask, aa, bb), 0);
          policy.spawn( f );

          if (end == nwork) break;
        }
        Kokkos::Experimental::wait( policy );
        
        const double dt = timer.seconds();

        if (!i) 
          dt_min = dt ;
        else 
          dt_min = (dt < dt_min ? dt : dt_min);
      }
      execution_space::fence();

      return dt_min ;
    }

  };

}

