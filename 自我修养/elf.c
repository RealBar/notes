int printf(const char* format, ...);

int global_init_var = 84;
int global_uinit_var ;
void func1(int i){
    printf("%d\n",i);
}

int main(void){
    static int static_var = 92;
    static int static_var2;
    
    int a  = 1;
    int b;

    func1(static_var2+static_var+a+b);
    return a;
}