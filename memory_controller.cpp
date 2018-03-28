#include "memory_controller.hpp"

namespace fool {

MemoryController::MemoryController()
	: m_handler(UNINITIALIZED), m_cpu_ptr(nullptr), m_size(0){

}

MemoryController::MemoryController(size_t size)
	: m_handler(UNINITIALIZED), m_cpu_ptr(nullptr), m_size(size){
}

MemoryController::~MemoryController(){
	if(m_cpu_ptr != nullptr){
		FoolFreeHost(m_cpu_ptr);
	}
}

inline void MemoryController::to_cpu(){
	switch (m_handler) {
	case UNINITIALIZED:
		FoolMallocHost(&m_cpu_ptr, m_size);
		fool_memset(m_cpu_ptr, 0, m_size);
		m_handler = AT_CPU;
		break;
	case AT_CPU:
		break;
	}
}

const void* MemoryController::cpu_data(){
	to_cpu();
	return (const void*)m_cpu_ptr;
}
void* MemoryController::mutable_cpu_data(){
	to_cpu();
	return m_cpu_ptr;
}

} // fool namespace
