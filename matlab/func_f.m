function f=func_f(y,a,b,n)
syms s;
f=ones(n,1);
for i=1:n
    f(i)=int(y*s^(i-1),s,a,b);
end