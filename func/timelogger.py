import pika
import json

conn_params = pika.ConnectionParameters('rabbit', 5672)
connection = pika.BlockingConnection(conn_params)
channel = connection.channel()

det_list = list()
img_list = list()
copy_list = list()
logs = list()

i = 0

while 1:
    status = channel.queue_declare(queue='time_logs')
    if status.method.message_count > 0:
        method_frame, header_frame, body = channel.basic_get('time_logs')
#        print(body)
        logs.append(body.decode())
        if method_frame:
            channel.basic_ack(method_frame.delivery_tag)
        else:
            print('No message returned')
        msg = json.loads(body)
#        print(msg)
        if msg['module'] == 'Detector' and 'Full' in msg['message']:
            det_list.append(float(msg['dt']))
            if len(det_list) % 10 == 0:
                print('Detector: ', len(det_list) // 10, sum(det_list[-10:]) / 10)
        if msg['module'] == 'ImgLoader' and 'Full' in msg['message']:
            img_list.append(float(msg['dt']))
            if len(img_list) % 10 == 0:
                print('ImgLoader: ', len(img_list) // 10, sum(img_list[-10:]) / 10)
#        if msg['module'] == 'ImgLoader' and 'Copied' in msg['message']:
#            copy_list.append(float(msg['dt']))
#            if len(copy_list) % 10 == 0:
#                print('Copy inside ImgLoader: ', len(copy_list) // 10, sum(copy_list[-10:]) / 10)

    i += 1
    if i % 500 == 0:
        with open('/code/logs.txt', 'a') as f:
            for log in logs:
                f.write(log+'\n')
        f.close()
        logs = list()
