extern int c;

int add(int a, int b);
int minus(int a, int b);

int main(){
    int aa = 123,bb  = 142134;
    int b = c - add(aa,bb)+minus(aa,bb);
    return b;
}