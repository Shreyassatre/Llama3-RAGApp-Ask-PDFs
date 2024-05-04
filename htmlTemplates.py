css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
img{
max-width: 3vw;
max-height: 3vw;
border-radius: 50%;
object-fit: cover;
}
'''

bot_template = '''
<div class="chat-message bot">
    <img src="https://s.yimg.com/ny/api/res/1.2/FPFrD5Y1vt5wtKG4y210Xg--/YXBwaWQ9aGlnaGxhbmRlcjt3PTk2MDtoPTUzODtjZj13ZWJw/https://media.zenfs.com/ko/am730_578/51de40fa2f652256835d2956eb5fdad2" />
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">   
    <img src="https://png.pngtree.com/element_our/png/20181206/users-vector-icon-png_260862.jpg" />
    <div class="message">{{MSG}}</div>
</div>
'''