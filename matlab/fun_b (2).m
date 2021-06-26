function w=fun_b(a,b,n,lamuda)
syms s t
w=ones(n,1);
for i=1:n
    x=fun_x(t);  
    p=fun_p(a,b,n);
    q=fun_q(a,b,n,lamuda);
    for tt=a:0.1:b
        q1=norm(subs(q,t,tt),Inf);
        p1=int(abs(exp(s*t)-kn),s,a,b);


   
   
    w(i)=fun_e(x,xn,a,b);
end
e