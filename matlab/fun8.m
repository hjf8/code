v=ones(1,9);
B=2*eye(10);
C=diag(v,1);
D=diag(v,-1);
A=B+C+D;
x0=ones(1,10)';
b=zeros(10,1);
b(1,1)=1;
x1=A\b;
t=[];
n=10;
for k=1:n
    for j=1:10
        x(j)=(b(j)-A(j,[1:j-1,j+1:10])*x0([1:j-1,j+1:10]))/A(j,j);
    end
    t=[t,x];
    x0=x';
end
disp('���̾�ȷ��:');
x1
disp('���̵���10����Ľ�:');
x
disp('����ÿ�ε����Ľ�:');
m=reshape(t',10,10)
g=[];
for l=1:10
    g=[g,m(:,l)-x1];
end
g%����ÿһ�м�ȥ��ȷ��
for h=1:10
    y(h)=norm(g(:,h));
end
y
p=[1:10];
plot(p,y,'*');




    