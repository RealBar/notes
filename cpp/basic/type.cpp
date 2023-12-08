#include<iostream>
#include<string>

class User{
private:
    std::string _name;
    int _age;
public:
    User() = default;
    ~User() = default;
    User(std::string name, int age):_name(name),_age(age){};
    void SetName(std::string name){
        _name = name;
    }
        std::string GetName() const {
        return _name;
    }
};


int main(){
    int a = 1;
    const int &b = a;
    a = 2;
    std::cout << b << std::endl;
    

    User u{"章三",12};

    int c = 1;
    int &d = c;
     int const &e = 2;

    return 0;
}

void ProcessUser(const User &u){
    // error: the object has type qualifiers that are not compatible with the member function "User::SetName"C/C++(1086)
    // type.cpp(31, 5): object type is: const User
    // u.SetName("李四");
    std::cout << u.GetName();
}