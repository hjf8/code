function pwxff(f,x0,x1,x2,d,n)
f=inline(f);
x(1)=x0;
x(2)=x1;
x(3)=x2;
w1=(f(x(2))-f(x(3)))/(x(2)-x(3));
t1=(f(x(1))-f(x(3)))/(x(1)-x(3));
t2=(f(x(1))-f(x(2)))/(x(1)-x(2));
w2=1/(x(1)-x(2))*(t1-t2);
w=w1+w2*(x(3)-x(2));
for k=3:n
    x(k+1)=x(k)-2*f(x(k))/(w+sqrt(w^2-4*f(x(k))*w2));
    if abs(x(k+1)-x(k))<d
        break
    end
    disp(sprintf('%d  %f',k,x(k+1)))
end
x=x(k+1)