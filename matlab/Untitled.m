max=100;
v=ones(1,9);
B=2*eye(10);
C=diag(v,1);
D=diag(v,-1);
A=B+C+D;
X0=ones(1,10)';
b=zeros(10,1);
b(1,1)=1;b(10,1)=1;
D1=diag(diag(A));dD=det(D1);
if dD==0
    disp('�ԽǾ���D���죬�޽�.')
else
    disp('�ԽǾ���D�����죬�н�.')
    iD=inv(D1);B2=iD;jX=A\b;%jx�Ǿ�ȷ��
    X=X0;
    for k=1:max
        X1=X+B2*(b-A*X);djwcX=norm(X1-X,Inf);
        if(djwcX<wucha)
            return
        else
            k,X1',k=k+1;X=X1;
        end
    end
    if(djwcX<wucha)|(xdwcX<wucha)
        disp('��ע�⣺��˹�����¶�������������A�ķֽ����D,U,L�;�ȷ��jX�ͽ��ƽ�X���£�')
    else
        disp('�������û�дﵽ�����ľ��ȣ����������ѳ�������������max,��ȷ��jX�͵�������X���£�')
        X=X';jX=jX'
    end
end
X=X';D,U,L,jX=jX'
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
save data3 A b m x1