from .registry import 

def till_intersection(person_bbox, camera_bbox):
    '''
    Checks whether person bbox lies into camera bbox.
    '''
    # TLBR
    if camera_bbox[1] <= person_bbox[1] and camera_bbox[3] >= person_bbox[3] and \
                    camera_bbox[2] <= person_bbox[2] and camera_bbox[0] >= person_bbox[0]:
        return True
    return False


def till_check(registry_entries, frame, till_camera_bboxes, timestamp, images, bb_fvs):
    '''Checks tills precense.
    Till camera bboxes -- dict {cam_id : bbox}
    Main camera ids for store 183: 86, 87.
    Partially 90, 91, 83 and 183 cameras also.
    '''
    for person in registry_entries:
        time_till_enter = registry_entries[person].time_till_enter
        time_till_exit = registry_entries[person].time_till_exit
        till_visited = registry_entries[person].till_visited

        for place in registry_entries[person]['places']:
            if place['cam_id'] in till_camera_bboxes:
                camera_bbox = till_camera_bboxes[place['cam_id']]
                is_intersected = till_intersection(place['bbox'], camera_bbox)
                if is_intersected and time_till_enter is None:
                    time_till_enter = place['timestamp']
                elif is_intersected and time_till_enter:
                    time_till_exit = place['timestamp']
                elif not is_intersected and time_till_enter is not None:
                    time_till_exit = place['timestamp']
                    if time_till_exit - time_till_enter >= 10:
                        till_visited = True
                        break
                    else:
                        time_till_enter = None
                        time_till_exit = None
            else:
                if time_till_exit and time_till_enter and time_till_exit - time_till_enter >= 10:
                    till_visited = True
                    break
                else:
                    time_till_enter = None
                    time_till_exit = None
                        
        if time_till_exit and time_till_enter and time_till_exit - time_till_enter >= 10:
            till_visited = True
        else:
            till_visited = None
        if till_visited:
            registry_entries[person].till_visited = till_visited
            registry_entries[person].time_till_enter = time_till_enter
            registry_entries[person].time_till_exit = time_till_exit
    return registry_entries
