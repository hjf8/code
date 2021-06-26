function x=gs(A,b)
%A=[1 2 3;3 4 5;6 7 8];b=[1;2;3];
n=size(A,1);
for i=1:n-1
    if(A(i,i)==0)
        t=min(find(A(i+1:n,1))+i);
        if t==[]
            disp('A «∆Ê“Ïæÿ’Û')
            return
        end
        m1=A(i,:);m2=b(i);
        A(i,:)=A(t,:);b(i)=b(t);
        A(t,:)=m1;b(t)=m2;
    end
    for j=i+1:n
        m=-A(j,i)/A(i,i);
        A(j,i)=0;
        A(j,i+1:n)=A(j,i+1:n)+m*A(i,i+1:n);
        b(j)=b(j)+m*b(i);
    end
end
x(n)=b(n)/A(n,n);
for i=n-1:-1:1
    x(i)=(b(i)-sum(x(i+1:n).*A(i,i+1:n)))/A(i,i);
end
x
        