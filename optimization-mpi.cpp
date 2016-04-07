/*
  Branch and bound algorithm to find the minimum of continuous binary
  functions using interval arithmetic.

  Sequential version

  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
*/

#include "interval.h"
#include "functions.h"
#include "minimizer.h"
#include "omp.h"
#include <chrono>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <stdexcept>
#include <string>

using namespace std;

int numproc, procrank;
double globalMin;

// Split a 2D box into four subboxes by splitting each dimension
// into two equal subparts
void split_box(const interval &x, const interval &y, interval &xl, interval &xr,
               interval &yl, interval &yr) {
  double xm = x.mid();
  double ym = y.mid();
  xl = interval(x.left(), xm);
  xr = interval(xm, x.right());
  yl = interval(y.left(), ym);
  yr = interval(ym, y.right());
}

// Branch-and-bound minimization algorithm
void minimize(itvfun f,           // Function to minimize
              const interval &x,  // Current bounds for 1st dimension
              const interval &y,  // Current bounds for 2nd dimension
              double threshold,   // Threshold at which we should stop splitting
              double &min_ub,     // Current minimum upper bound
              minimizer_list &ml) // List of current minimizers
{
  interval fxy = f(x, y);

  if (fxy.left() > min_ub) { // Current box cannot contain minimum?
    return;
  }

  if (fxy.right() < min_ub) { // Current box contains a new minimum?
    min_ub = fxy.right();
    // Discarding all saved boxes whose minimum lower bound is
    // greater than the new minimum upper bound*
    if (procrank == 0) {
      auto discard_begin = ml.lower_bound(minimizer{0, 0, min_ub, 0});
      ml.erase(discard_begin, ml.end());
    }
  }

  // Checking whether the input box is small enough to stop searching.
  // We can consider the width of one dimension only since a box
  // is always split equally along both dimensions
  // We 
  if ((x.width() * numproc) <= threshold) { //At the begining, x dimension 
  											//is divided by the number of proc.
    // We have potentially a new minimizer
    if (procrank == 0) {
      ml.insert(minimizer{x, y, fxy.left(), fxy.right()});
    }
    return;
  }

  // The box is still large enough => we split it into 4 sub-boxes
  // and recursively explore them
  interval xl, xr, yl, yr;
  split_box(x, y, xl, xr, yl, yr);
  minimize(f, xl, yl, threshold, min_ub, ml);
  minimize(f, xl, yr, threshold, min_ub, ml);
  minimize(f, xr, yl, threshold, min_ub, ml);
  minimize(f, xr, yr, threshold, min_ub, ml);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &procrank);

  cout.precision(16);
  // By default, the currently known upper bound for the minimizer is +oo
  double min_ub = numeric_limits<double>::infinity();
  // List of potential minimizers. They may be removed from the list
  // if we later discover that their smallest minimum possible is
  // greater than the new current upper bound
  minimizer_list minimums;
  // Threshold at which we should stop splitting a box
  double precision;

  // Name of the function to optimize
  string choice_fun;

  // The information on the function chosen (pointer and initial box)
  opt_fun_t fun;

  bool good_choice;

  // MPI Initialiazing
  int choice_size;

  if (procrank == 0) {
    // Asking the user for the name of the function to optimize
    do {
      good_choice = true;

      cout << "Which function to optimize?\n";
      cout << "Possible choices: ";
      for (auto fname : functions) {
        cout << fname.first << " ";
      }
      cout << endl;
      cin >> choice_fun;

      try {
        fun = functions.at(choice_fun);
      } catch (out_of_range) {
        cerr << "Bad choice" << endl;
        good_choice = false;
      }
    } while (!good_choice);

    choice_size = choice_fun.size();
  }

  MPI_Bcast(&choice_size, 1, MPI_INT, 0, MPI_COMM_WORLD);//Broadcast
  
  char buf[choice_size + 1];

  if (procrank == 0)
    strcpy(buf, choice_fun.c_str());

  MPI_Bcast(&buf, choice_size + 1, MPI_CHAR, 0, MPI_COMM_WORLD);//Broadcast
  choice_fun.assign(buf, choice_size);

  if (procrank != 0) {
    fun = functions.at(choice_fun);
  }

  if (procrank == 0) {
    // Asking for the threshold below which a box is not split further
    cout << "Precision? ";
    cin >> precision;
  }

  MPI_Bcast(&precision, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double dist = (abs(fun.x.left()) + abs(fun.x.right())) / numproc;

  double left = fun.x.left() + dist * procrank;
  double right = left + dist;
  cout << procrank << ":" << left << ":" << right << endl;
  interval inter(left, right);

  double min_final;

  auto start_time = chrono::high_resolution_clock::now();

  minimize(fun.f, inter, fun.y, precision, min_ub, minimums);

  MPI_Reduce(&min_ub, &min_final, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

  auto end_time = chrono::high_resolution_clock::now();

  if (procrank == 0) {
	  cout << "Upper bound for minimum: " << min_final << endl;
	  cout << "Temps: "
		   << chrono::duration_cast<chrono::seconds>(end_time - start_time).count()
		   << ":";
	  cout << chrono::duration_cast<chrono::microseconds>(end_time - start_time)
		          .count()
		   << "  secondes" << endl;
  }
  MPI_Finalize();
}
