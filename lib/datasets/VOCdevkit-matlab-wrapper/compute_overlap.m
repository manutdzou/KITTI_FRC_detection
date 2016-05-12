function overlap=compute_overlap(bb_pred,bb_target)
a=(bb_pred(3)-bb_pred(1)+1)*(bb_pred(4)-bb_pred(2)+1);
b=(bb_target(3)-bb_target(1)+1)*(bb_target(4)-bb_target(2)+1);
bb_overlap=[max(bb_pred(1),bb_target(1)),max(bb_pred(2),bb_target(2)),min(bb_pred(3),bb_target(3)),min(bb_pred(4),bb_target(4))];
iw=bb_overlap(3)-bb_overlap(1)+1;
ih=bb_overlap(4)-bb_overlap(2)+1;
if iw>0&ih>0
    overlap=iw*ih/(a+b-iw*ih);
else
    overlap=-inf;
end

