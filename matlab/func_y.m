function y=func_y(x,k,a,b,lambda)
syms t s;
y=subs(lambda*x-int(x*k,t,a,b),t,s);
