function e=degenerate_kernel_methods(a,b,n,lambda)
e=ones(n,1);
syms s t;% defining two symbolic variables
for i=1:n
    k=func_k(s,t);% kernel function
    x=func_x(t);  
    y=func_y(x,k,a,b,lambda);
    f=func_f(y,a,b,i);
    c=solve_c(f,b,lambda,i);
    xn=solve_xn(y,c,lambda,i);
    e(i)=err(x,xn,a,b); % error
end
disp(e);
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


    