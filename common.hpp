#ifndef COMMON_HPP
#define COMMON_HPP
#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>
#include <climits>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <map>

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
	template class classname<float>; \
	template class classname<double>

namespace fool {
	using boost::shared_ptr;

	using std::vector;
	using std::set;
	using std::map;
	using std::string;
	using std::ostringstream;
	using std::fstream;
}
#endif // COMMON_HPP
