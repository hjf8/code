function p=fun_p(a,b,n)
syms s t
kn=0;
for i=0:3
    kn=kn+s^i*t^i/factorial(i);
end
p=int(abs(exp(s*t)-kn)s,a,b)

 