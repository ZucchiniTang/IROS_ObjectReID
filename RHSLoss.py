class RHSLoss(nn.Module):
    def __init__(self, T=1.0):
        super(RHSLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.T = T

    def forward(self, q, k, centers, label):
        # q: torch.Size([32, 2048]);  k: torch.Size([32, 2048]);  label: torch.Size([32])
        batchSize = q.shape[0]  # 32
        N = q.size(0)
        
        dist_centers = torch.matmul(q, centers.transpose(1, 0)) # size: ([32, 238]); min():-0.2972;  max():0.3190
        mask = torch.ones_like(dist_centers).scatter_(1, label.unsqueeze(1), 0.)
        res_dist_centers = dist_centers[mask.bool()].view(batchSize, -1)  # size: ([32, 237])
        
        mat_sim = torch.matmul(q, k.transpose(0, 1))   # mat_sim: torch.Size([32, 32])
        mat_eq = label.expand(N, N).eq(label.expand(N, N).t()).float()   # mat_eq: torch.Size([32, 32])
        loss = []
        for b in range(batchSize):  # 0~31
            logit = mat_sim[b]
            target = mat_eq[b]
            
            res_dist_center = res_dist_centers[b] # [237]
            
            pos_logit = torch.masked_select(logit, target > 0.5)
            pos_logit_min = pos_logit.min()
            neg_logit = torch.masked_select(logit, target < 0.5)
            hard_center_logit_max = res_dist_center.max()
            
            ll = []
            ## Fig. 4: Step 1 - Comparing relative distance
            if hard_center_logit_max >= pos_logit_min:
              
                ## Step 2 - Selecting hard positive samples and hard negative center
                hard_pos_logit = torch.masked_select(pos_logit, pos_logit<=hard_center_logit_max)
                hard_neg_center = torch.masked_select(res_dist_center, res_dist_center>=pos_logit_min)
                
                ## Step 3 - Relative hard samples learning
                for p in hard_pos_logit:
                    out = torch.cat((p.unsqueeze(0), neg_logit, hard_neg_center)) / self.T
                    triple_dist = F.log_softmax(out,dim=0)
                    targets = torch.zeros([batchSize]).cuda().long()
                    triple_dist_ref = torch.zeros_like(triple_dist).scatter_(0, targets, 1)
                    l = (- triple_dist_ref * triple_dist).sum()
                    ll.append(l)
                ll = torch.mean(torch.stack(ll))
            else:
                out = torch.cat((pos_logit_min.unsqueeze(0), neg_logit, hard_center_logit_max.unsqueeze(0))) / self.T
                triple_dist = F.log_softmax(out,dim=0)
                targets = torch.zeros([batchSize]).cuda().long()
                triple_dist_ref = torch.zeros_like(triple_dist).scatter_(0, targets, 1)
                ll = (- triple_dist_ref * triple_dist).sum()
                    
            loss.append(ll)
        loss = torch.mean(torch.stack(loss))
        return loss
