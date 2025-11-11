def generate_site_and_geom_tags():
    base_pos = [-0.0115 ,-0.0425 ,0.0181]
    offset_outer = [0, 0.0025, 0]
    offset_inner = [0, 0, -0.0025]
    
    geom_tags = []
    site_tags = []
    
    for i in range(8):
        pos_outer = [round(base_pos[j] + i * offset_outer[j], 4) for j in range(3)]
        for k in range(4):
            pos_inner = [round(pos_outer[j] + k * offset_inner[j], 4) for j in range(3)]
            site_name = f"mid_tip_{i}_{k}"
            geom_name = f"mid_tip_geom_{i}_{k}"
            
            site_tag = f'<site name="{site_name}" pos="{" ".join(map(str, pos_inner))}" size="0.0005 0.001 0.001" rgba="1 0 1 1" type="box"/>'
            geom_tag = f'<geom name="{geom_name}" pos="{" ".join(map(str, pos_inner))}" size="0.0005 0.001 0.001" rgba="1 1 1 0" type="box"/>'
            
            geom_tags.append(geom_tag)
            site_tags.append(site_tag)
    
    return geom_tags, site_tags

if __name__ == "__main__":
    geom_tags, site_tags = generate_site_and_geom_tags()
    
    for tag in geom_tags:
        print(tag)
    
    for tag in site_tags:
        print(tag)


# def generate_touch_tags(finger_names):
#     touch_tags = []
    
#     for finger in finger_names:
#         for i in range(8):
#             for k in range(4):
#                 touch_name = f"{finger}_tip_{i}_{k}"
#                 touch_tag = f'<touch name="{touch_name}" site="{touch_name}"/>'
#                 touch_tags.append(touch_tag)
    
#     return touch_tags

# if __name__ == "__main__":
#     finger_names = ["thumb", "index", "mid", "ring"]
#     touch_tags = generate_touch_tags(finger_names)
    
#     for tag in touch_tags:
#         print(tag)

