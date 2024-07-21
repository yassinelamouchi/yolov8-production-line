import math

class Tracker:
    def __init__(self, counting_area):
        self.center_points = {}
        self.id_count = 0
        self.counting_area = counting_area  # (x, y, width, height)
        self.counted_ids = set()

    def is_within_area(self, rect):
        x, y, x2, y2 = rect
        area_x, area_y, area_w, area_h = self.counting_area
        return (x >= area_x and y >= area_y and x2 <= (area_x + area_w) and y2 <= (area_y + area_h))

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, x2, y2 = rect
            cx = (x + x2) // 2
            cy = (y + y2) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, x2, y2, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, x2, y2, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids
