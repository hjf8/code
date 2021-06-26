function[x]=gauss(A,b)
n=size(A,1);%����A������
x=zeros(1,n);
for i=1:n-1
    if(A(i,i)==0)
        t=min(find(A(i+1:n,1)~=0)+i);%�Ҿ���A�д�i+1�п�ʼԪ�ز�Ϊ���Ԫ�����ڵĵط�����������С��Ԫ��ȡ����
        if(isempty(t))
            disp('A matrix is signular');
            return
        end;
        tmpA=A(i,:);
        tmpb=b(i);
        A(i,:)=A(t,:);%����t�����i�н���
        b(i)=b(t);
        A(t,:)=tmpA;
        b(t)=tmpb;
    end;
    for j=i+1:n
        m=-A(j,i)/A(i,i);
        A(j,i)=0;
        A(j,i+1:n)=A(j,i+1:n)+m*A(i,i+1:n);%Guass��Ԫ���Ĳ���
        b(j)=b(j)+m*b(i);
    end
end
x(n)=b(n)/A(n,n);%�����һ��Ԫ�������
for i=n-1:-1:1%��n-1��1
    x(i)=(b(i)-sum(x(i+1:n).*A(i,i+1:n)))/A(i,i);
end