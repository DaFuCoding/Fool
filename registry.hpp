#ifndef FOOL_REGISTRY_HPP
#define FOOL_REGISTRY_HPP

#include <memory>
#include <mutex>
#include <map>
#include <functional>

template <class SrcType, class ObjectType, class... Args>
class Registry {
public:
	typedef std::function<std::unique_ptr<ObjectType> (Args ...)> Creator;

	Registry() : registry_() {}

	void Register(const SrcType& key, Creator creator) {
		std::lock_guard<std::mutex> lock(register_mutex_);
		if (registry_.count(key) != 0) {
			std::printf("Key already registered.\n");
			std::exit(1);
		}
		registry_[key] = creator;
	}

	std::unique_ptr<ObjectType> Create(const SrcType& key, Args ... args) {
		if (registry_.count(key) == 0) {
			return nullptr;
		}
		return registry_[key](args...);
	}
private:
	std::map<SrcType, Creator> registry_;
	std::mutex register_mutex_;
//DISABLE_COPY_AND_ASSIGN(Registry);
};

template <class SrcType, class ObjectType, class... Args>
class Registerer {
 public:
	Registerer(const SrcType& key,
						 Registry<SrcType, ObjectType, Args...>* registry,
						 typename Registry<SrcType, ObjectType, Args...>::Creator creator) {
		registry->Register(key, creator);
	}
	template <class DerivedType>
	static std::unique_ptr<ObjectType> DefaultCreator(Args ... args) {
		return std::unique_ptr<ObjectType>(new DerivedType(args...));
	}
};
class Foo{
public:
	explicit Foo(int x) { std::cout << "FOO" << x ;}
};
class Bar:public Foo{
public:
	explicit Bar(int x): Foo(x){ std::cout << "Bar" << x ;}
};

Registry<std::string, Foo, int>* FooRegistry(){
	static Registry<std::string, Foo, int>* registry =
			new Registry<std::string, Foo, int>();
	return registry;
}
typedef Registerer<std::string, Foo, int> RegistererFooRegistry;

//static RegistererFooRegistry g_FooRegistry110("asd",);


#endif // FOOL_REGISTRY_HPP
