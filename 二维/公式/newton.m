function [k,x,wucha,yx]=newton(x0,tol)
k=1;
yx1=fun(x0);
yx2=fun1(x0);
x1=x0-yx1/yx2;
while abs(x1-x0)>tol
    x0=x1;
    yx1=fun(x0);
    yx2=fun1(x0);
    k=k+1;
    x1=x1-yx1/yx2;
end
k;
x=x1;
wucha=abs(x1-x0)/2;
yx=fun(x);
end