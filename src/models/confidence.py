import torch
import torch.nn.functional as F

######################################################################
# Probability Map Extraction
######################################################################
def confidence_estimation(prob_volume, fused_maps, start_depths, depth_intervals, device, method="swr"):
    ### 4-NEIGHBOR SUMMATION ###
    if (method=="nsum"):
        batch_size, channels, height, width = fused_maps.shape
        _, _, depth, _, _ = prob_volume.shape

        d_coords = torch.unsqueeze((fused_maps - depth_start) / depth_interval,2)
        d_coords_left0 = torch.clamp(torch.floor(d_coords).type(torch.long),0,depth-1)
        d_coords_left1 = torch.clamp( d_coords_left0 - 1, 0, depth-1)
        d_coords_right0 = torch.clamp(torch.ceil(d_coords).type(torch.long), 0, depth-1)
        d_coords_right1 = torch.clamp(d_coords_right0 + 1, 0, depth-1)

        prob_map_left0 = torch.gather(prob_volume, dim=2, index=d_coords_left0)
        prob_map_left1 = torch.gather(prob_volume, dim=2, index=d_coords_left1)
        prob_map_right0 = torch.gather(prob_volume, dim=2, index=d_coords_right0)
        prob_map_right1 = torch.gather(prob_volume, dim=2, index=d_coords_right1)

        prob_maps = prob_map_left0 + prob_map_left1 + prob_map_right0 + prob_map_right1
        prob_maps = torch.reshape(prob_maps, (batch_size, 1, height, width))



    ### ROUNDED LOCATION OF MAX PROBABILITY ###
    elif(method=="single"):
        batch_size, height, width = fused_maps.shape
        _, depth, _, _ = prob_volume.shape

        d_coords = torch.unsqueeze((fused_maps - start_depths) / depth_intervals, 1)
        d_coords = torch.clamp(torch.round(d_coords).type(torch.long),0,depth-1)
        prob_maps = torch.gather(prob_volume, dim=1, index=d_coords)
        prob_maps = torch.reshape(prob_maps, (batch_size, 1, height, width))



    ### PADDED (zero-padding) 4-NEIGHBOR SUMMATION w/ Range Width###
    elif(method=="pns"):
        batch_size, height, width = fused_maps.shape

        # padding pattern (pad last dim by (0,0), second-to-last by (0,0), third-to-last by (1,1)... why is pytorch padding like this....)
        padding = (0,0,0,0,1,1)
        padded_prob_volume = F.pad(prob_volume, padding)

        batch_size, depth, height, width = padded_prob_volume.shape

        # stepped version of gauss
        stepped_gauss = torch.zeros(depth)
        stepped_gauss[1:depth-1] = 1.0
        stepped_gauss = stepped_gauss.reshape(1,depth,1,1).repeat(batch_size,1,height,width).to(device)
        padded_prob_volume = torch.mul(stepped_gauss, padded_prob_volume)

        # get local values
        d_coords = torch.unsqueeze((fused_maps - start_depths) / depth_intervals, 1) + 1
        d_coords_left0 = torch.clamp(torch.floor(d_coords).type(torch.long), 0, depth-1)
        d_coords_left1 = torch.clamp((d_coords_left0 - 1).type(torch.long), 0, depth-1)
        d_coords_right0 = torch.clamp(torch.ceil(d_coords).type(torch.long), 0, depth-1)
        d_coords_right1 = torch.clamp((d_coords_right0 + 1).type(torch.long), 0, depth-1)

        prob_map_left0 = torch.gather(padded_prob_volume, dim=1, index=d_coords_left0)
        prob_map_left1 = torch.gather(padded_prob_volume, dim=1, index=d_coords_left1)
        prob_map_right0 = torch.gather(padded_prob_volume, dim=1, index=d_coords_right0)
        prob_map_right1 = torch.gather(padded_prob_volume, dim=1, index=d_coords_right1)

        prob_maps = prob_map_left0 + prob_map_left1 + prob_map_right0 + prob_map_right1
        prob_maps = torch.reshape(prob_maps, (batch_size, 1, height, width))


    ### PADDED (zero-padding) 4-NEIGHBOR SUMMATION w/ Search Window Radius###
    elif(method=="pns_swr"):
        batch_size, height, width = fused_maps.shape

        # padding pattern (pad last dim by (0,0), second-to-last by (0,0), third to last by (1,1)... why is pytorch padding like this....)
        padding = (0,0,0,0,1,1)
        padded_prob_volume = F.pad(prob_volume, padding)

        batch_size, depth, height, width = padded_prob_volume.shape

        # stepped version of gauss
        stepped_gauss = torch.zeros(depth)
        stepped_gauss[1:depth-1] = 1.0
        stepped_gauss = stepped_gauss.reshape(1,depth,1,1).repeat(batch_size,1,height,width).to(device)
        padded_prob_volume = torch.mul(stepped_gauss, padded_prob_volume)

        # get local values
        d_coords = torch.unsqueeze((fused_maps - start_depths) / depth_intervals, 1) + 1
        d_coords_left = torch.clamp(torch.floor(d_coords).type(torch.long), 0, depth-1)
        d_coords_right = torch.clamp(torch.ceil(d_coords).type(torch.long), 0, depth-1)

        prob_map_left = torch.gather(padded_prob_volume, dim=1, index=d_coords_left)
        prob_map_right = torch.gather(padded_prob_volume, dim=1, index=d_coords_right)

        prob_maps = prob_map_left + prob_map_right
        prob_maps = torch.reshape(prob_maps, (batch_size, 1, height, width))
        
        # damp proportional to plane interval size (corresponds to search window radius)
        max_int = torch.max(depth_intervals)
        min_int = torch.min(depth_intervals)
        norm_intervals = (depth_intervals-min_int) / (max_int-min_int)
        conf_scale = (1-norm_intervals).unsqueeze(1).type(torch.float32)
        prob_maps = torch.mul(conf_scale, prob_maps)


    ### Search Window Radius ###
    elif(method=="swr"):
        batch_size, height, width = fused_maps.shape

        # confidence proportional to plane interval size (corresponds to search window radius)
        max_int = torch.max(depth_intervals)
        min_int = torch.min(depth_intervals)
        norm_intervals = (depth_intervals-min_int) / (max_int-min_int)
        conf_scale = (1-norm_intervals).unsqueeze(1).type(torch.float32)
        prob_maps = conf_scale


    ### PADDED (replication-padding) PKRN ###
    elif(method=="pkrn"):
        batch_size, height, width = fused_maps.shape

        # padding pattern (pad last dim by (0,0), second-to-last by (0,0), third to last by (1,1)... why is pytorch padding like this....)
        padding = (0,0,0,0,1,1)
        padded_prob_volume = F.pad(prob_volume, padding, mode='replicate')

        _, depth, _, _ = padded_prob_volume.shape

        # adding +1 to the coords because of the padding
        d_coords = torch.unsqueeze((fused_maps - start_depths) / depth_intervals,1) + 1
        d_coords_left = torch.clamp(torch.floor(d_coords).type(torch.long), 0, depth)
        d_coords_right = torch.clamp(torch.ceil(d_coords).type(torch.long), 0, depth)

        prob_map_left = torch.gather(padded_prob_volume, dim=1, index=d_coords_left)
        prob_map_right = torch.gather(padded_prob_volume, dim=1, index=d_coords_right)

        prob_maps = torch.cat((prob_map_left, prob_map_right), dim=1)

        (max_vals, max_inds) = torch.max(prob_maps, dim=1, keepdims=True)

        min_inds = torch.sub(1, max_inds)
        min_vals = torch.gather(prob_maps, dim=1, index=min_inds)

        prob_maps = torch.div(min_vals, max_vals)
        prob_maps = torch.sub(1, prob_maps)
        prob_maps = torch.reshape(prob_maps, (batch_size, 1, height, width))

        # damp proportional to plane interval size (corresponds to search window radius)
        max_int = torch.max(depth_intervals)
        norm_intervals = depth_intervals / max_int
        conf_scale = (1-norm_intervals).unsqueeze(1).type(torch.float32)
        prob_maps = torch.mul(conf_scale, prob_maps)


    return prob_maps
