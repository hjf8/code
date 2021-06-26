function u=LU(A,b,p,q)
%q��ʾ�ϴ���p��ʾ�´���
n=size(A);
%��״�����LU�ֽ�
for k=1:n-1
    for i=k+1:min(k+p,n)
        A(i,k)=A(i,k)/A(k,k);
    end
    for j=k+1:min(k+q,n)
        for i=k+1:min(k+p,n)
            A(i,j)=A(i,j)-A(i,k)*A(k,j);
        end
    end
end
L=eye(n)+tril(A,-1);
U=triu(A);
%��״���Ƿ��������
%��ǰ��ȥ��������ʽ
for j=1:n
    for i=j+1:min(j+p,n)
        b(i)=b(i)-L(i,j)*b(j);
    end
end
%�����ȥ��;����ʽ
for j=n:-1:1
    b(j)=b(j)/U(j,j);
    for i=max(1,j-p):j-1
        b(i)=b(i)-U(i,j)*b(j);
    end
end
u=b;