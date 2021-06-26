function [x]=gradient(A,b,x0,wucha,N)
x=x0;
r=b-A*x;
P=r;
%t=zeros(1,10);
for i=1:N
    z=A*r;
    alpha=(r'*r)/(z'*r);
    x=x+alpha*r;  %X1=X0+alpha0*P0
    s=(A*P)'*P;
    r=r-alpha*z;
    e=norm(r);%残差范数
    %t(i)=e;%迭代残差
    if (e<wucha)
        %t
        %plot(1:10,t,'r')
        %xlabel('迭代次数'),ylabel('error')
        %title('gradient')
        return 
    end
    belta=-((A*r)'*P)/s;
    P=r+belta*P;  %P1=r1+belta0*P0
    i=i+1;
end
x;


