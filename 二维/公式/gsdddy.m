function X=gsdddy(A,b,X0,P,wucha,max1)
D=diag(diag(A));U=-triu(A,1);
L=-tril(A,-1);dD=det(D);
if dD==0
    disp('�ԽǾ���D���죬�޽�.')
else
    disp('�ԽǾ���D�����죬�н�.')
    iD=inv(D-L);B2=iD*U;f2=iD*b;jX=A\b;
    X=X0;[n m]=size(A);
    for k=1:max1
        X1=B2*X+f2;djwcX=norm(X1-X,P);
        xdwcX=djwcX/(norm(X,P)+eps);
        if(djwcX<wucha)|(xdwcX<wucha)
            return
        else
            k,X1',k=k+1;X=X1;
        end
    end
    if(djwcX<wucha)|(xdwcX<wucha)
        disp('��ע�⣺��˹�����¶�������������A�ķֽ����D,U,L�;�ȷ��jX�ͽ��ƽ�X���£�')
    else
        disp('�������û�дﵽ�����ľ��ȣ����������ѳ�������������max1,��ȷ��jX�͵�������X���£�')
        X=X';jX=jX'
    end
end
X=X';D,U,L,jX=jX'