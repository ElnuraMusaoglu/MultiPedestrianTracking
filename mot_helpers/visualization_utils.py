import cv2
from os.path import join

def show_tracks(tracks, current_frame, idx, write_image=None, file_path=None):
    for track in tracks:
        roi = track.roi
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

        cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)

        cv2.putText(current_frame,
                    str(track.track_id),
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('tracking_single', current_frame)
        c = cv2.waitKey(1) & 0xFF

        if c == 27 or c == ord('q'):
            break

    if write_image and file_path:
        cv2.imwrite(join(file_path, str(idx) + '.jpg'), current_frame)


    # cv2.destroyAllWindows()