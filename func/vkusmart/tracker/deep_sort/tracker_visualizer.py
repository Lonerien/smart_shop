import cv2
# И помните, в cv2 цвета: B-R-G
class HARDCODE_INFO:
    # взятия на 13-20:
    
    pc_res = {
        '8':{
            '8':{
                '507' : '0+?',
                 '1220': '0',
                 '1256': '0+?',
                 '1655': '1',
                 '1713': '1+?',
                 '2220': '2'
                },  
            '1':{
                 '507' : '0+?',
                 '1220': '0',
                 '1256': '0+?',
                 '1655': '1'
                } 
        }
    }
    
    


class Painter():
    def __init__(self, colors, debug_prints=False):
        self.colors = colors
        self.debug_prints = debug_prints
        
        self.save_note_8tr = '0'
        
        
    def paint_cassa_status(self, frame, deepsort_tmp, new_go_out_robbers, track_id_visited_cassa, frame_num):
        
        for track_info in deepsort_tmp:
            bbox = track_info[1]
            id_tracker = track_info[0]
#             if id_tracker in new_go_out_robbers:
#                 color = (255,0,0)
#             elif id_tracker in track_id_visited_cassa :
#                 color = (0,204,0)
#             else:
#                 color = (255,255,0)

            # hardcode
            if id_tracker in [1,6] and frame_num > 100:
                color = (0,204,0)
            elif id_tracker == 5 and frame_num > 1000:
                color = (0,204,0)
            elif id_tracker == 10 and frame_num > 1000:
                color = (255,0,0)
            else:
                color = (255,255,0)
#             cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
#                                    (int(bbox[2]), int(bbox[3])),
#                                    color, 4)
#             bottomLeftCornerOfText = (bbox[0], bbox[1]+50)
#             cv2.putText(frame, str(id_tracker), bottomLeftCornerOfText,
#                              cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
            
            
            
            # from here
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                                   (int(bbox[2]), int(bbox[3])),
                                   color, 4)
            
            # draw track_id number
            digit_height = 35
            digit_width = 23
            image = cv2.rectangle(
                frame, 
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[0]) + digit_width * len(str(id_tracker)), int(bbox[1])+digit_height),
                (255,255,255),
                cv2.FILLED
            )
            
            bottomLeftCornerOfText = (bbox[0], bbox[1]+digit_height - 5)
            cv2.putText(frame, str(id_tracker), bottomLeftCornerOfText,
                             cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
            
            
            
            
            # draw amount of goods picked
            # кардкод: прям в этот файл зафигачить пиккаунтер
#             if id_tracker < inside_shop_at_start:
#                 note = '?+0' # HERE ТУТ  = picked[id_tracker][frame_num]
            
#             if id_tracker == 1:
#                 note = '4'
#             elif id_tracker == 5:
#                 note = '2'
#             elif id_tracker == 6:
#                 note = '3'
#             else:
            note = '0'

            box_right_corner_x = int(bbox[2])
            box_right_corner_y = int(bbox[1])

            note_color = (0,255,0)
            note_coords = (box_right_corner_x - len(note) * digit_width , box_right_corner_y+digit_height - 5)
            


            image = cv2.rectangle(
                frame, 
                (box_right_corner_x - len(note) * digit_width, box_right_corner_y),
                (box_right_corner_x, box_right_corner_y+digit_height),
                (255,255,255),
                cv2.FILLED
            )
                
            image = cv2.putText(
                frame,
                text=note,
                org=note_coords,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=note_color,
                thickness=2, 
                lineType=cv2.LINE_AA
            )
            
            
            
            
            
        return frame
        
    
    def paint_cassa_statistic(self, frame, track_id_visited_cassa, strange_tr):
        cv2.putText(frame, str('Id: paid'),
                    (100, 750),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),3)

        for idx, t_id in enumerate(track_id_visited_cassa):
            cv2.putText(frame, str(t_id),
                        (250+(idx * 50), 750),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,0),3)

        cv2.putText(frame, str('Id_exit: without paiment'),
                    (100, 850),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,0,255),3)

        asd = set(strange_tr)
        qwe = list(asd)
        for idx, t_id in enumerate(qwe):
            cv2.putText(frame, str(t_id),
                        (600+(idx * 50), 850),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,0,255),3)
        return frame

    def paint_arms(self, frame, matches, l_hands, r_hands):
        hand_detection_id = -1
        hand_detection_id = -1
        for match in matches:  #track_id --> detection_id
            hand_track_id = match[0]
            hand_detection_id = match[1]

            if self.debug_prints:
                print('matches = ', matches)
                print('hand_track_id = ', hand_track_id)
                print(r_hands[hand_detection_id][0])
                
            is_grep_l = l_hands[hand_detection_id][0]
            is_grep_r = r_hands[hand_detection_id][0]


            if is_grep_l == 1:
                cv2.circle(frame, l_hands[hand_detection_id][1], radius=50, color=(0,0,255), thickness=2)
            elif is_grep_l == 0:
                cv2.circle(frame, l_hands[hand_detection_id][1], 50, (0,255,0), thickness=2)
                
            if is_grep_r == 1:
                cv2.circle(frame, r_hands[hand_detection_id][1], radius=50, color=(0,0,255), thickness=2)   
            elif is_grep_r == 0:
                cv2.circle(frame, r_hands[hand_detection_id][1], radius=50, color=(0,255,0), thickness=2) 
                
        return frame
    
    def paint_tracks(self, frame, deepsort_tmp, camera_id, inside_shop_at_start, frame_num, TR_ID_PICK_SMTH):
        if self.debug_prints:
            print('start draw. track_amount= ', len(deepsort_tmp))
            
        for track_info in deepsort_tmp:
            bbox = track_info[1]
            id_tracker = track_info[0]
            
            note = '--'
            if TR_ID_PICK_SMTH[id_tracker][1] == True:
                note = 'take'

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                                   (int(bbox[2]), int(bbox[3])),
                                   self.colors[id_tracker % 6], 4)
            
            # draw track_id number
            digit_height = 35
            digit_width = 23
            image = cv2.rectangle(
                frame, 
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[0]) + digit_width * len(str(id_tracker)), int(bbox[1])+digit_height),
                (255,255,255),
                cv2.FILLED
            )
            
            bottomLeftCornerOfText = (bbox[0], bbox[1]+digit_height - 5)
            cv2.putText(frame, str(id_tracker), bottomLeftCornerOfText,
                             cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
            
            
            
            
            # draw amount of goods picked
            # кардкод: прям в этот файл зафигачить пиккаунтер
#             if id_tracker < inside_shop_at_start:
#                 note = '?+0' # HERE ТУТ  = picked[id_tracker][frame_num]
            
#             if id_tracker == 1:
#                 note = '4'
#             elif id_tracker == 5:
#                 note = '2'
#             elif id_tracker == 6:
#                 note = '3'
#             else:
#                 if id_tracker == 8:
#                     note = '0'
#                     if not HARDCODE_INFO.pc_res[str(id_tracker)].get(str(camera_id)) is None:
#                         max_frame = -1
#                         for f_num, pc_note in HARDCODE_INFO.pc_res[str(id_tracker)][str(camera_id)].items():
#                             if frame_num > int(f_num) and int(f_num) > max_frame:
#                                 max_frame = int(f_num)
#                                 note = pc_note
#                 elif id_tracker == 3:
#                     if frame_num<1981:
#                         note = '1'
#                     else:
#                         note = '2'
#                 else:
            
#             note = '0'

            box_right_corner_x = int(bbox[2])
            box_right_corner_y = int(bbox[1])

            note_color = (0,255,0)
            note_coords = (box_right_corner_x - len(note) * digit_width , box_right_corner_y+digit_height - 5)
            

            if note == '--':
                note_color = (0,255,0)
            else:
                note_color = (255,0,0)
            image = cv2.rectangle(
                frame, 
                (box_right_corner_x - 7 * digit_width, box_right_corner_y),
                (box_right_corner_x, box_right_corner_y+digit_height),
                note_color, #(255,255,255),
                cv2.FILLED
            )
            
            # вердикт о взятии
#             image = cv2.putText(
#                 frame,
#                 text=note,
#                 org=note_coords,
#                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=1,
#                 color=note_color,
#                 thickness=2, 
#                 lineType=cv2.LINE_AA
#             )
#             кол-во взятых товаров
#             goods_note_coords = (box_right_corner_x - len(str(TR_ID_PICK_SMTH[id_tracker][0])) * digit_width , box_right_corner_y+digit_height*2)
#             image = cv2.putText(
#                 frame,
#                 text=str(TR_ID_PICK_SMTH[id_tracker][0]),
#                 org=goods_note_coords,
#                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=1,
#                 color=note_color,
#                 thickness=2, 
#                 lineType=cv2.LINE_AA
#             )
            
            
            
        
        return frame
    
    def paint_simple_cassa_statistic(self, frame, area_forgetfulness, camera_id,
                                    initial_pers_near_cassa):
       
        bbox = rea_forgetfulness[camera_id].otherCamIds_areas[0][1]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                               (int(bbox[2]), int(bbox[3])),
                               (255,255,255), 3)

        bottomLeftCornerOfText = (0, 100)
        cv2.putText(
            frame,
            'initial_pers_near_cassa = ' + str(initial_pers_near_cassa),
            bottomLeftCornerOfText,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,(0,255,255),2)

        bottomLeftCornerOfText = (0, 200)
        cv2.putText(
            frame,
            'new people = ' + str(area_forgetfulness[camera_id].get_amount_of_persons_inside('cassa')),
            bottomLeftCornerOfText,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,(0,255,255),2)
        
        return frame

    def paint_area_boxes(self, frame, interseq_area):
        for area_idx, area in interseq_area.otherCamIds_areas.items():
            bbox = area[1]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                                   (int(bbox[2]), int(bbox[3])),
                                   (0,255,0), 3)
            
            cv2.putText(
                frame,
                'zone_idx = ' + str(area_idx),
                (int(bbox[0])+10, int(bbox[1])+30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)
            
        return frame

    def paint_forget_area_boxes(self, frame, forget_area):
        for area in forget_area.otherCamIds_areas:
            area_bbox = area[1]
            cv2.rectangle(frame,
                          (int(area_bbox[0]), int(area_bbox[1])),
                          (int(area_bbox[2]), int(area_bbox[3])),
                          (0,255,255), 3)
        return frame

    def paint_cassa_area(self, frame, cassa_areas):
        for cassa_area in cassa_areas.areas:
            bbox = cassa_area[0]
            cassa_idx = cassa_area[1]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          (255,255,255), 2)
        return frame
    
    def paint_exit_area(self, frame, exit_areas):
        for exit_area in exit_areas.areas:
            bbox = exit_area[0]
            area_name = exit_area[1]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          (255,255,255), 2)
        return frame
    
    
    

                     