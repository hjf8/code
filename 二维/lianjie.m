function lianjie = lianjie(node_x,node_y,nel)
%�����������Ľڵ���Ŀ���͵�Ԫ���ɶ�
%������Ӿ���ÿ����Ԫ�漰�Ľڵ�ı��
%node_x=5;node_y=5;nel=3;
xn = 1:(node_x*node_y);%���һ��������
A = reshape(xn,node_x,node_y);%ͬ��״���

for i = 1:(node_x-1)*(node_y-1)%���ε�Ԫ
    x = rem(i,node_x-1);%��ʾ��ԪΪx��������ڼ���   
    if x == 0
        x = node_x-1;
    end
    y = ceil(i/(node_x-1));%����ȡ��,y��������ڼ���
    a = A(x:x+1,y:y+1);%���С������ֱ�˾������Ӿ���
    a_vec = a(:);
    lianjie(2*i-1:2*i,1:nel) = [a_vec([1 4 3])';a_vec([4 1 2])'];
end



