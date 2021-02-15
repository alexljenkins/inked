def define_env(env):
    @env.macro
    def news(info, end_date):

        site = env.conf["site_url"].strip("/")

        return f"""
        <div id="news" end_date="{end_date}" class="admonition info" style="display:none;">
            <p class="admonition-title">Info</p>
            {info}
        </div>
<script>
var news = document.getElementById('news')
if (new Date() < new Date(news.getAttribute('end_date'))) {{
    news.style.display = 'block'
}}
</script>
        """.strip()
