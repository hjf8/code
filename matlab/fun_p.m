%function p=fun_p(a,b,n)
a=0;b=1;n=1;
syms s t
kn=0;
for i=0:n
    kn=kn+s^i*t^i/factorial(i);
end
kn
p1=abs(exp(s*t)-kn);
%for tt=a:0.1:b
%     p=vpa(norm(subs(p1,t,tt),Inf),20);
%end
%p
 

 