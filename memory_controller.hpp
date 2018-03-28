#ifndef MEMORY_CONTROLLER_HPP
#define MEMORY_CONTROLLER_HPP
#include <stdio.h>
#include <memory>

#include <glog/logging.h>

namespace fool{


inline void fool_memset(void* X, const int value, const size_t size){
	memset(X, value, size);
}

// adapt CUDA
inline void FoolMallocHost(void** ptr, size_t size){
#ifndef CPU_ONLY
	LOG(ERROR) << "Don't support GPU." ;
	return;
#endif
	*ptr = malloc(size);
}
inline void FoolFreeHost(void* ptr){
#ifndef CPU_ONLY
	return;
#endif
	free(ptr);
}

// Allocate memory when needed
class MemoryController{
public:
	MemoryController();
	explicit MemoryController(size_t size);
	~MemoryController();
	void to_cpu();

	const	void* cpu_data();
	void set_cpu_data(void* data);
	void* mutable_cpu_data();

	size_t size()	{return m_size;}
	enum HandlerType{UNINITIALIZED, AT_CPU};
	HandlerType m_handler;
	void* m_cpu_ptr;
	size_t m_size;

};

}

#endif // MENCONTROLLER_HPP
