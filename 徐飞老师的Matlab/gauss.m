function[x]=gauss(A,b)
n=size(A,1);%矩阵A的行数
x=zeros(1,n);
for i=1:n-1
    if(A(i,i)==0)
        t=min(find(A(i+1:n,1)~=0)+i);%找矩阵A中从i+1行开始元素不为零的元素所在的地方，将其中最小的元素取出来
        if(isempty(t))
            disp('A matrix is signular');
            return
        end;
        tmpA=A(i,:);
        tmpb=b(i);
        A(i,:)=A(t,:);%将第t行与第i行交换
        b(i)=b(t);
        A(t,:)=tmpA;
        b(t)=tmpb;
    end;
    for j=i+1:n
        m=-A(j,i)/A(i,i);
        A(j,i)=0;
        A(j,i+1:n)=A(j,i+1:n)+m*A(i,i+1:n);%Guass消元法的步骤
        b(j)=b(j)+m*b(i);
    end
end
x(n)=b(n)/A(n,n);%把最后一个元素求出来
for i=n-1:-1:1%从n-1到1
    x(i)=(b(i)-sum(x(i+1:n).*A(i,i+1:n)))/A(i,i);
end