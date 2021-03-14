from iqradre.detect.ops import boxes as boxes_ops
from iqradre.detect.ops import box_ops

def build_annoset(text_list, boxes):
    boxes_list = box_ops.batch_box_coordinate_to_xyminmax(boxes, to_int=True).tolist()    
    annoset = [{'text':t, "bbox": b}  for t,b in zip(text_list, boxes_list)] 
    return annoset