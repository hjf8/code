function xn=solve_xn(y,c,lambda,n)
syms t;
m=0;
for i=0:n-1
    m=m+c(i+1)*t^i/factorial(i);
end
xn=(y+m)/lambda;