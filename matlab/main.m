function e=main(a,b,n,lamuda)
e=ones(n,1);
syms s t;% defining two symbolic variables
for i=1:n
    %k=fun_k(s,t);% kernel function
    x=fun_x(t);  
    y=fun_y(x,a,b,lamuda);
    f=fun_f(y,a,b,n);
    c=solve_c(f,b,lamuda,n);
    xn=solve_xn(y,c,lamuda,n);
    e(i)=err(x,xn,a,b); % error
end
e;
% the graph of the errors
if n==10
    figure(1)
    tt=a:0.1:b;
    plot(tt,vpa(subs(subs(x-xn,s,t),t,tt),20),'r--')
    xlabel('t'), ylabel('error')
end
figure(2)
plot(1:10,-log10(e),'o')
xlabel('degree n'), ylabel('-log10 of error')
