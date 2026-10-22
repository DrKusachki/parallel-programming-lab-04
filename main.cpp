#include <iostream>
#include <omp.h>
#include <chrono>
#include <thread>

//const double PI = 3.1415'9265'3589'7932'3846;
const double PI = std::acos(-1.0);
const int num_steps = 100'000'000;
const double step = 1.0 / static_cast<double>(num_steps);
const int threads = 4;
double sum[threads];
const int test_num = 50;

double Pi()
{
  double x = 0.0, sum_ = 0.0;
  for ( int i = 0; i < num_steps; ++i )
  {
    x = ( (double)i + 0.5 ) * step;
    sum_ += 4.0 / ( 1.0 + x * x );
  }
  return sum_ * step;
}

void ParallelPi(int thread_num)
{
  double x = 0.0, sum_ = 0.0;
  for ( int i = thread_num; i < num_steps; i += threads )
  {
    x = ( (double)i + 0.5 ) * step;
    sum_ += 4.0 / ( 1.0 + x * x );
  }
  sum[thread_num] = sum_;
}

double OMPPi()
{
  double x = 0.0, sum_ = 0.0;
  #pragma omp parallel num_threads(threads) private(x)
  {
    #pragma omp for schedule(static) reduction(+:sum_)
      for ( int i = 0; i < num_steps; ++i )
      {
        x = ( (double)i + 0.5 ) * step;
        sum_ += 4.0 / ( 1.0 + x * x );
      }
  }
  return sum_ * step;
}

int main()
{
  std::chrono::milliseconds def_pi_total_time{ 0 };
  std::chrono::milliseconds par_pi_total_time{ 0 };
  std::chrono::milliseconds omp_pi_total_time{ 0 };
  std::chrono::milliseconds duration{ 0 };
  using hrc = std::chrono::high_resolution_clock;
  hrc::time_point start, end;
  
  double errorDef = 0.0, errorPar = 0.0, errorOMP = 0.0;

  const char * line = "---------------------------------\n";

  for ( int i = 0; i < test_num; ++i )
  {
    printf( "Test #%i:\n\n", i + 1 );
    {
      start = hrc::now();
      double result = Pi();
      end = hrc::now();
      duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start );
      def_pi_total_time += duration;
      errorDef += ::abs( result - PI );
      printf( "Default pi evaluation algorithm: %.16lf\nTime taken: %lld ms\nError: %.4e\n",
              result, duration.count(), ::abs( result - PI ) );
    }
    printf( line );
    {
      start = hrc::now();
      std::thread threads_[threads] = {
        std::thread(ParallelPi, 0),
        std::thread(ParallelPi, 1),
        std::thread(ParallelPi, 2),
        std::thread(ParallelPi, 3)
      };
      double result = 0.0;
      for ( int i = 0; i < threads; ++i )
      {
        threads_[i].join();
        result += sum[i] * step;
      }
      end = hrc::now();
      duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start );
      par_pi_total_time += duration;
      errorPar += ::abs( result - PI );
      printf( "Parallel pi evaluation algorithm: %.16lf\nTime taken: %lld ms\nError: %.4e\n",
              result, duration.count(), ::abs( result - PI ) );
    }
    printf( line );
    {
      start = hrc::now();
      double result = OMPPi();
      end = hrc::now();
      duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start );
      omp_pi_total_time += duration;
      errorOMP += ::abs( result - PI );
      printf( "Parallel pi evaluation algorithm via OMP: %.16lf\nTime taken: %lld ms\nError: %.4e\n",
              result, duration.count(), ::abs( result - PI ) );
    }
    printf( "%s\n", line);
  }

  printf( "Results: \n" );
  printf( "Average time for the default pi evaluation algoritm: %lld\nAverage error for the default pi evaluation algorithm: %.4e\n", 
          def_pi_total_time.count() / test_num, errorDef / (double)test_num );
  printf( "Average time for the parallel pi evaluation algoritm: %lld\nAverage error for the parallel pi evaluation algorithm: %.4e\n", 
          par_pi_total_time.count() / test_num, errorPar / (double)test_num );
  printf( "Average time for the openMP pi evaluation algoritm: %lld\nAverage error for the openMP pi evaluation algorithm: %.4e\n", 
          omp_pi_total_time.count() / test_num, errorOMP / (double)test_num );
  return 0;
}