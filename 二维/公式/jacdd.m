function X=jacdd(A,b,X0,P,wucha,max1)
[n,m]=size(A);
for j=1:m
    a(j)=sum(abs(A(:,j)))-2*(abs(A(j,j)));
end
for i=1:n
    if a(i)>=0
        disp('��ע�⣬A�����ϸ�Խ�ռ�ţ����ſɱȵ�����һ������')
        return
    end
end
if a(i)<0
    disp('��ע�⣺A�ϸ�Խ�ռ�ţ��˷�������Ψһ�����ſɱȵ�������')
end
for k=1:max1
    k
    for j=1:m
        X(j)=(b(j)-A(j,[1:j-1,j+1:m])*X0([1:j-1,j+1:m]))/A(j,j);
    end
    X,djwcX=norm(X'-X0,P);xdwcX=djwcX/(norm(X',P)+eps);X0=X';X1=A\b;
    if (djwcX>wucha)&(xdwcX<wucha)
        disp('��ע�⣺�ſɱȵ����������˷�����ľ�ȷ��jX�ͽ��ƽ�X���£�')
        return
    end
end
if (djwcX>wucha)&(xdwcX>wucha)
    disp('��ע�⣺�ſɱȵ��������Ѿ���������������max1')
end
a,X=X;jX=X1',